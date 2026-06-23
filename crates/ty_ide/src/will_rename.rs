use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, Ordering};

use ruff_db::files::{File, system_path_to_file};
use ruff_db::source::source_text;
use ruff_db::system::{SystemPath, SystemPathBuf};
use ruff_python_ast::token::{TokenKind, Tokens};
use ruff_python_ast::{
    self as ast, AnyNodeRef,
    visitor::source_order::{SourceOrderVisitor, TraversalSignal},
};
use ruff_text_size::{Ranged, TextRange};
use rustc_hash::{FxHashMap, FxHashSet};
use ty_module_resolver::{Module, ModuleName, file_to_module};
use ty_project::Db;
use ty_python_semantic::types::Type;
use ty_python_semantic::{
    HasType, ImportAliasResolution, ResolvedDefinition, SemanticModel, definitions_for_name,
};

/// A text edit to apply before renaming a file or directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRenameEdit {
    file: File,
    range: TextRange,
    new_text: String,
}

impl FileRenameEdit {
    /// Decomposes this edit into its target file, range, and replacement text.
    pub fn into_parts(self) -> (File, TextRange, String) {
        (self.file, self.range, self.new_text)
    }
}

/// A file or directory rename received from the client.
#[derive(Debug, Clone)]
pub struct PathRename {
    kind: PathRenameKind,
    old_path: SystemPathBuf,
    new_path: SystemPathBuf,
}

impl PathRename {
    /// Creates a file rename from `old_path` to `new_path`.
    pub fn file(old_path: SystemPathBuf, new_path: SystemPathBuf) -> Self {
        Self {
            kind: PathRenameKind::File,
            old_path,
            new_path,
        }
    }

    /// Creates a directory rename from `old_path` to `new_path`.
    pub fn directory(old_path: SystemPathBuf, new_path: SystemPathBuf) -> Self {
        Self {
            kind: PathRenameKind::Directory,
            old_path,
            new_path,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum PathRenameKind {
    File,
    Directory,
}

struct ModuleMove {
    old: ModuleName,
    new: ModuleName,
    anchor: File,
}

struct ModuleMoves {
    by_old_name: FxHashMap<ModuleName, ModuleName>,
    anchors: FxHashSet<File>,
}

impl ModuleMoves {
    fn needles(&self) -> FxHashSet<&str> {
        self.by_old_name
            .keys()
            .map(ModuleName::last_component)
            .collect()
    }

    fn rewrite(&self, name: &ModuleName) -> Option<ModuleName> {
        let (old_prefix, new_prefix) = name.ancestors().find_map(|old_prefix| {
            self.by_old_name
                .get(&old_prefix)
                .map(|new_prefix| (old_prefix, new_prefix))
        })?;
        if &old_prefix == name {
            Some(new_prefix.clone())
        } else {
            let suffix = name.relative_to(&old_prefix)?;
            ModuleName::from_components(new_prefix.components().chain(suffix.components()))
        }
    }
}

struct ImportingModule {
    name_after_move: ModuleName,
    is_package: bool,
    moved: bool,
}

/// Compute edits for a batch of file and directory renames.
///
/// The batch is first normalized to semantic module moves. Every project file
/// is then visited once, retaining the syntax needed to render each edit.
/// Returns no edits if the batch cannot be represented without overlapping
/// edits or splitting a single `from` import across multiple parent modules.
pub fn will_rename_paths(db: &dyn Db, renames: &[PathRename]) -> Vec<FileRenameEdit> {
    let Some(moves) = module_moves(db, renames) else {
        return Vec::new();
    };
    if moves.by_old_name.is_empty() {
        return Vec::new();
    }

    let needles = moves.needles();
    let files = db.project().files(db);
    // A renamed file can be excluded from the project index but is still editable
    // at its old URI while `willRenameFiles` is applied.
    let extra_anchors: Vec<_> = moves
        .anchors
        .iter()
        .copied()
        .filter(|anchor| !files.contains(anchor))
        .collect();
    let result = std::sync::Mutex::new(Vec::new());
    let supported = AtomicBool::new(true);

    {
        let db = Db::dyn_clone(db);
        let result = &result;
        let supported = &supported;
        let moves = &moves;
        let needles = &needles;

        rayon::scope(move |scope| {
            for file in (&files).into_iter().chain(extra_anchors) {
                let db = Db::dyn_clone(&*db);
                scope.spawn(move |_| {
                    let db = &*db;
                    let source = source_text(db, file);
                    let parsed = ruff_db::parsed::parsed_module(db, file);
                    let module = parsed.load(db);
                    let importing_module = file_to_module(db, file).map(|module| {
                        let old_name = module.name(db);
                        if let Some(new_name) = moves.rewrite(old_name) {
                            ImportingModule {
                                name_after_move: new_name,
                                is_package: module.kind(db).is_package(),
                                moved: true,
                            }
                        } else {
                            ImportingModule {
                                name_after_move: old_name.clone(),
                                is_package: module.kind(db).is_package(),
                                moved: false,
                            }
                        }
                    });

                    let contains_needle = module.tokens().iter().any(|token| {
                        token.kind() == TokenKind::Name
                            && source
                                .get(usize::from(token.start())..usize::from(token.end()))
                                .is_some_and(|name| needles.contains(name))
                    });
                    let requires_import_context_scan = importing_module
                        .as_ref()
                        .is_some_and(|importing_module| importing_module.moved)
                        && contains_relative_import(module.tokens());
                    if !contains_needle && !requires_import_context_scan {
                        return;
                    }

                    let model = SemanticModel::new(db, file);
                    let mut edits = Vec::new();
                    let mut file_supported = true;
                    let mut finder = ModuleMoveFinder {
                        model: &model,
                        tokens: module.tokens(),
                        source: source.as_str(),
                        moves,
                        importing_module: importing_module.as_ref(),
                        edits: &mut edits,
                        supported: &mut file_supported,
                    };
                    AnyNodeRef::from(module.syntax()).visit_source_order(&mut finder);

                    if file_supported {
                        result.lock().unwrap().extend(edits);
                    } else {
                        supported.store(false, Ordering::Relaxed);
                    }
                });
            }
        });
    }

    if !supported.load(Ordering::Relaxed) {
        return Vec::new();
    }

    let mut edits = result.into_inner().unwrap();
    edits.sort_by(|left, right| {
        left.file
            .cmp(&right.file)
            .then_with(|| left.range.start().cmp(&right.range.start()))
            .then_with(|| left.range.end().cmp(&right.range.end()))
            .then_with(|| left.new_text.cmp(&right.new_text))
    });
    edits.dedup();
    // Overlapping edits cannot be represented safely in an LSP workspace edit.
    if edits.windows(2).any(|edits| {
        edits[0].file == edits[1].file
            && (edits[0].range.start() == edits[1].range.start()
                || edits[0].range.end() > edits[1].range.start())
    }) {
        return Vec::new();
    }
    edits
}

fn module_moves(db: &dyn Db, renames: &[PathRename]) -> Option<ModuleMoves> {
    let mut by_old_name = FxHashMap::default();
    let mut anchors = FxHashSet::default();

    for rename in renames {
        let Some(module_move) = module_move(db, rename) else {
            continue;
        };
        anchors.insert(module_move.anchor);
        match by_old_name.entry(module_move.old) {
            Entry::Vacant(entry) => {
                entry.insert(module_move.new);
            }
            Entry::Occupied(entry) => {
                if entry.get() != &module_move.new {
                    return None;
                }
            }
        }
    }

    Some(ModuleMoves {
        by_old_name: by_old_name
            .into_iter()
            .filter(|(old, new)| old != new)
            .collect(),
        anchors,
    })
}

fn module_move(db: &dyn Db, rename: &PathRename) -> Option<ModuleMove> {
    let (old_file, old_module, new_name) = match rename.kind {
        PathRenameKind::File => {
            if !matches!(rename.new_path.extension(), Some("py" | "pyi")) {
                return None;
            }
            let old_file = system_path_to_file(db, &rename.old_path).ok()?;
            let old_module = file_to_module(db, old_file)?;
            let new_name =
                infer_new_module_name(db, &rename.old_path, &rename.new_path, &old_module)?;
            (old_file, old_module, new_name)
        }
        PathRenameKind::Directory => {
            let old_file = package_init_file(db, &rename.old_path)?;
            let old_module = file_to_module(db, old_file)?;
            let new_name = infer_directory_module_name(db, &rename.new_path, &old_module)?;
            (old_file, old_module, new_name)
        }
    };

    Some(ModuleMove {
        old: old_module.name(db).clone(),
        new: new_name,
        anchor: old_file,
    })
}

fn package_init_file(db: &dyn Db, directory: &SystemPath) -> Option<File> {
    let init_path = directory.join("__init__.py");
    let init_pyi_path = directory.join("__init__.pyi");

    system_path_to_file(db, &init_path)
        .or_else(|_| system_path_to_file(db, &init_pyi_path))
        .ok()
}

struct ModuleMoveFinder<'a, 'db> {
    model: &'a SemanticModel<'db>,
    tokens: &'a Tokens,
    source: &'a str,
    moves: &'a ModuleMoves,
    importing_module: Option<&'a ImportingModule>,
    edits: &'a mut Vec<FileRenameEdit>,
    supported: &'a mut bool,
}

impl<'ast> SourceOrderVisitor<'ast> for ModuleMoveFinder<'_, '_> {
    fn enter_node(&mut self, node: AnyNodeRef<'ast>) -> TraversalSignal {
        match node {
            AnyNodeRef::StmtImport(import) => {
                self.check_import(import);
                TraversalSignal::Skip
            }
            AnyNodeRef::StmtImportFrom(import_from) => {
                self.check_import_from(import_from);
                TraversalSignal::Skip
            }
            AnyNodeRef::ExprName(name) => {
                if self.check_module_expression(ast::ExprRef::Name(name)) {
                    TraversalSignal::Skip
                } else {
                    TraversalSignal::Traverse
                }
            }
            AnyNodeRef::ExprAttribute(attribute) => {
                if self.check_module_expression(ast::ExprRef::Attribute(attribute)) {
                    TraversalSignal::Skip
                } else {
                    TraversalSignal::Traverse
                }
            }
            _ => TraversalSignal::Traverse,
        }
    }
}

impl ModuleMoveFinder<'_, '_> {
    fn check_import(&mut self, import: &ast::StmtImport) {
        for alias in &import.names {
            let Some(module) = self.model.resolve_module(Some(alias.name.as_str()), 0) else {
                continue;
            };
            let old_name = module.name(self.model.db());

            if let Some(new_name) = self.moves.rewrite(old_name) {
                self.add_edit(alias.name.range, new_name.as_str());
            }
        }
    }

    fn check_import_from(&mut self, import: &ast::StmtImportFrom) {
        let Ok(old_parent) =
            ModuleName::from_import_statement(self.model.db(), self.model.file(), import)
        else {
            return;
        };

        let default_parent = self
            .moves
            .rewrite(&old_parent)
            .unwrap_or_else(|| old_parent.clone());
        let mut statement_parent = None;
        let mut alias_edits = Vec::new();

        for alias in &import.names {
            let alias_parent = if let Some(Type::ModuleLiteral(module)) =
                alias.inferred_type(self.model)
            {
                let old_name = module.module(self.model.db()).name(self.model.db());
                if is_direct_submodule_import(&old_parent, alias, old_name)
                    && let Some(new_name) = self.moves.rewrite(old_name)
                {
                    let Some(new_parent) = new_name.parent() else {
                        if import.names.len() == 1 {
                            let alias = alias
                                .asname
                                .as_ref()
                                .map_or_else(String::new, |asname| format!(" as {asname}"));
                            self.add_edit(import.range, format!("import {new_name}{alias}"));
                        } else {
                            *self.supported = false;
                        }
                        return;
                    };

                    if alias.name.as_str() != new_name.last_component() {
                        alias_edits.push((alias.name.range, new_name.last_component().to_string()));
                    }
                    new_parent
                } else {
                    default_parent.clone()
                }
            } else {
                default_parent.clone()
            };

            if statement_parent
                .as_ref()
                .is_some_and(|parent| parent != &alias_parent)
            {
                *self.supported = false;
                return;
            }
            statement_parent.get_or_insert(alias_parent);
        }

        let statement_parent = statement_parent.unwrap_or(default_parent);
        self.add_import_from_module_edit(import, &statement_parent);
        for (range, new_name) in alias_edits {
            self.add_edit(range, new_name);
        }
    }

    fn check_module_expression(&mut self, expression: ast::ExprRef<'_>) -> bool {
        let Some(Type::ModuleLiteral(module)) = expression.inferred_type(self.model) else {
            return false;
        };
        let old_name = module.module(self.model.db()).name(self.model.db());
        let Some((root, expression_name)) = module_expression_identity(self.model, expression)
        else {
            return false;
        };
        if &expression_name != old_name {
            return false;
        }
        let Some(new_name) = self.moves.rewrite(old_name) else {
            return false;
        };
        let Some(Type::ModuleLiteral(root_module)) = root.inferred_type(self.model) else {
            return false;
        };
        let old_root = root_module.module(self.model.db()).name(self.model.db());
        let new_root = self
            .moves
            .rewrite(old_root)
            .unwrap_or_else(|| old_root.clone());

        let root_name = root.id.as_str();
        let uses_imported_name = (root_name == old_root.first_component()
            || root_name == old_root.last_component())
            && root_uses_imported_module_name(self.model, root);
        let replacement = if uses_imported_name && root_name == old_root.first_component() {
            new_name.as_str().to_string()
        } else if uses_imported_name && root_name == old_root.last_component() {
            render_from_root(new_root.last_component(), &new_root, &new_name)
        } else {
            render_from_root(root_name, &new_root, &new_name)
        };

        self.add_edit(expression.range(), replacement);
        true
    }

    fn add_import_from_module_edit(&mut self, import: &ast::StmtImportFrom, new_name: &ModuleName) {
        let Some(range) = import_from_module_range(self.tokens, import) else {
            return;
        };
        let replacement = render_import_from_module(import, new_name, self.importing_module);
        self.add_edit(range, replacement);
    }

    fn add_edit(&mut self, range: TextRange, new_text: impl Into<String>) {
        let new_text = new_text.into();
        let source_range = usize::from(range.start())..usize::from(range.end());
        if self.source.get(source_range) == Some(new_text.as_str()) {
            return;
        }
        self.edits.push(FileRenameEdit {
            file: self.model.file(),
            range,
            new_text,
        });
    }
}

/// Returns whether `root` is bound directly by an unaliased module import.
///
/// Explicit aliases and other bindings keep their local spelling when the
/// module moves. Unaliased imports change the bound name with the import and
/// therefore require matching edits at their use sites.
fn root_uses_imported_module_name(model: &SemanticModel<'_>, root: &ast::ExprName) -> bool {
    let definitions = definitions_for_name(
        model,
        root.id.as_str(),
        root.into(),
        ImportAliasResolution::PreserveAliases,
    );
    !definitions.is_empty()
        && definitions
            .iter()
            .all(|definition| matches!(definition, ResolvedDefinition::Module(_)))
}

fn render_from_root(root: &str, root_name: &ModuleName, full_name: &ModuleName) -> String {
    if root_name == full_name {
        root.to_string()
    } else if let Some(suffix) = full_name.relative_to(root_name) {
        format!("{root}.{suffix}")
    } else {
        full_name.as_str().to_string()
    }
}

fn module_expression_identity<'a>(
    model: &SemanticModel<'_>,
    expression: ast::ExprRef<'a>,
) -> Option<(&'a ast::ExprName, ModuleName)> {
    let mut attributes = Vec::new();
    let root = expression_root_and_attributes(expression, &mut attributes)?;
    let Type::ModuleLiteral(module) = root.inferred_type(model)? else {
        return None;
    };
    let root_name = module.module(model.db()).name(model.db());
    let name = ModuleName::from_components(root_name.components().chain(attributes))?;
    Some((root, name))
}

fn expression_root_and_attributes<'a>(
    expression: ast::ExprRef<'a>,
    attributes: &mut Vec<&'a str>,
) -> Option<&'a ast::ExprName> {
    match expression {
        ast::ExprRef::Name(name) => Some(name),
        ast::ExprRef::Attribute(attribute) => {
            let root = expression_chain_root_and_attributes(&attribute.value, attributes)?;
            attributes.push(attribute.attr.as_str());
            Some(root)
        }
        _ => None,
    }
}

fn expression_chain_root_and_attributes<'a>(
    expression: &'a ast::Expr,
    attributes: &mut Vec<&'a str>,
) -> Option<&'a ast::ExprName> {
    match expression {
        ast::Expr::Name(name) => Some(name),
        ast::Expr::Attribute(attribute) => {
            let root = expression_chain_root_and_attributes(&attribute.value, attributes)?;
            attributes.push(attribute.attr.as_str());
            Some(root)
        }
        _ => None,
    }
}

fn is_direct_submodule_import(
    parent: &ModuleName,
    alias: &ast::Alias,
    module_name: &ModuleName,
) -> bool {
    module_name.parent().as_ref() == Some(parent)
        && module_name.last_component() == alias.name.as_str()
}

fn import_from_module_range(tokens: &Tokens, import: &ast::StmtImportFrom) -> Option<TextRange> {
    let mut after_from = false;
    let mut first = None;
    let mut last = None;

    for token in tokens.in_range(import.range) {
        match token.kind() {
            TokenKind::From => after_from = true,
            TokenKind::Import if after_from => break,
            TokenKind::Dot | TokenKind::Ellipsis | TokenKind::Name if after_from => {
                first.get_or_insert(token.start());
                last = Some(token.end());
            }
            _ => {}
        }
    }

    Some(TextRange::new(first?, last?))
}

fn contains_relative_import(tokens: &Tokens) -> bool {
    let mut after_from = false;

    for token in tokens {
        match token.kind() {
            TokenKind::From => after_from = true,
            TokenKind::Dot | TokenKind::Ellipsis if after_from => return true,
            TokenKind::Import if after_from => after_from = false,
            _ => {}
        }
    }

    false
}

fn render_import_from_module(
    import: &ast::StmtImportFrom,
    new_name: &ModuleName,
    importing_module: Option<&ImportingModule>,
) -> String {
    if import.level == 0 {
        return new_name.as_str().to_string();
    }

    let Some(importing_module) = importing_module else {
        return new_name.as_str().to_string();
    };
    let level = if importing_module.is_package {
        import.level.saturating_sub(1)
    } else {
        import.level
    };
    let Some(new_base) = importing_module
        .name_after_move
        .ancestors()
        .nth(level as usize)
    else {
        return new_name.as_str().to_string();
    };

    let relative = if new_name == &new_base {
        Some(String::new())
    } else {
        new_name
            .relative_to(&new_base)
            .map(|relative| relative.as_str().to_string())
    };
    let Some(relative) = relative else {
        return new_name.as_str().to_string();
    };

    let mut rendered = ".".repeat(import.level as usize);
    rendered.push_str(&relative);
    rendered
}

/// Infer the new module name for a renamed directory from its search path.
fn infer_directory_module_name(
    db: &dyn Db,
    new_dir: &SystemPath,
    old_module: &Module<'_>,
) -> Option<ModuleName> {
    let search_path = old_module.search_path(db)?.as_system_path()?;
    let new_relative = new_dir.strip_prefix(search_path).ok()?;
    ModuleName::from_components(
        new_relative
            .components()
            .map(|component| component.as_str()),
    )
}

/// Infer the new module name from old/new file paths.
///
/// For same-directory renames, replaces the last component of the old module
/// name using [`ModuleName::parent()`]. For cross-directory moves, derives the
/// new module name from the module's search path.
fn infer_new_module_name(
    db: &dyn Db,
    old_path: &SystemPath,
    new_path: &SystemPath,
    old_module: &Module<'_>,
) -> Option<ModuleName> {
    let new_stem = new_path.file_stem()?;
    let old_stem = old_path.file_stem()?;
    if new_stem == "__init__" || old_stem == "__init__" {
        return None;
    }

    let old_module_name = old_module.name(db);

    if old_path.parent() == new_path.parent() {
        return if let Some(parent) = old_module_name.parent() {
            ModuleName::from_components(parent.components().chain(std::iter::once(new_stem)))
        } else {
            ModuleName::new(new_stem)
        };
    }

    // Cross-directory move: derive the full module name from the search path.
    let search_path = old_module.search_path(db)?.as_system_path()?;
    let new_relative = new_path.strip_prefix(search_path).ok()?;
    let parent = new_relative.parent()?;
    ModuleName::from_components(
        parent
            .components()
            .map(|component| component.as_str())
            .chain(std::iter::once(new_stem)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruff_db::files::system_path_to_file;
    use ruff_db::source::source_text;
    use ruff_db::system::{DbWithWritableSystem, SystemPathBuf};
    use ruff_python_ast::PythonVersion;
    use ty_project::{ProjectMetadata, TestDb};

    #[test]
    fn rename_simple_import() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("consumer.py", "import old_module\n\nprint(old_module.x)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import new_module\n\nprint(new_module.x)\n");
    }

    #[test]
    fn rename_from_import() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("consumer.py", "from old_module import x\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.py"),
        );

        assert_eq!(edits.len(), 1);

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from new_module import x\n");
    }

    #[test]
    fn rename_no_edits_for_unrelated_files() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("other.py", "y = 2\n"),
            ("consumer.py", "import other\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.py"),
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_multiple_consumers() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("consumer1.py", "import old_module\n"),
            ("consumer2.py", "from old_module import x\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.py"),
        );

        assert_eq!(edits.len(), 2);
    }

    #[test]
    fn rename_nonexistent_file() {
        let db = create_test_db(&[("consumer.py", "import something\n")]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("nonexistent.py"),
            SystemPath::new("new_name.py"),
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_package_submodule() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/old_sub.py", "x = 1\n"),
            ("consumer.py", "from pkg.old_sub import x\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/old_sub.py"),
            SystemPath::new("pkg/new_sub.py"),
        );

        assert_eq!(edits.len(), 1);

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from pkg.new_sub import x\n");
    }

    #[test]
    fn rename_relative_import() {
        let db = create_test_db(&[
            (
                "pkg/__init__.py",
                "from .ner_model_port import NERModelPort as NERModelPort\n",
            ),
            ("pkg/ner_model_port.py", "class NERModelPort: ...\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/ner_model_port.py"),
            SystemPath::new("pkg/ner_model.py"),
        );

        assert_eq!(edits.len(), 1);

        let init = system_path_to_file(&db, "pkg/__init__.py").unwrap();
        let result = apply_edits(&db, &edits, init);
        assert_eq!(
            result,
            "from .ner_model import NERModelPort as NERModelPort\n"
        );
    }

    #[test]
    fn rename_relative_import_deep_package() {
        let db = create_test_db(&[
            ("qu/__init__.py", ""),
            ("qu/domain/__init__.py", ""),
            (
                "qu/domain/port/__init__.py",
                concat!(
                    "from .ai_model_port import AIModelPort as AIModelPort\n",
                    "from .ai_model_port import AIModelResult as AIModelResult\n",
                    "from .cache_port import CachePort as CachePort\n",
                    "from .dictionary_port import DictionaryPort as DictionaryPort\n",
                    "from .embed_model_port import EmbedModelPort as EmbedModelPort\n",
                    "from .ner_model_port import NERModelPort as NERModelPort\n",
                ),
            ),
            (
                "qu/domain/port/ai_model_port.py",
                "class AIModelPort: ...\nclass AIModelResult: ...\n",
            ),
            ("qu/domain/port/cache_port.py", "class CachePort: ...\n"),
            (
                "qu/domain/port/dictionary_port.py",
                "class DictionaryPort: ...\n",
            ),
            (
                "qu/domain/port/embed_model_port.py",
                "class EmbedModelPort: ...\n",
            ),
            (
                "qu/domain/port/ner_model_port.py",
                "class NERModelPort: ...\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("qu/domain/port/ner_model_port.py"),
            SystemPath::new("qu/domain/port/ner_model.py"),
        );

        assert_eq!(edits.len(), 1);

        let init = system_path_to_file(&db, "qu/domain/port/__init__.py").unwrap();
        let result = apply_edits(&db, &edits, init);
        let expected = concat!(
            "from .ai_model_port import AIModelPort as AIModelPort\n",
            "from .ai_model_port import AIModelResult as AIModelResult\n",
            "from .cache_port import CachePort as CachePort\n",
            "from .dictionary_port import DictionaryPort as DictionaryPort\n",
            "from .embed_model_port import EmbedModelPort as EmbedModelPort\n",
            "from .ner_model import NERModelPort as NERModelPort\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn rename_relative_import_multiple_lines() {
        let db = create_test_db(&[
            (
                "pkg/__init__.py",
                concat!(
                    "from .ai_model_port import AIModelPort as AIModelPort\n",
                    "from .ai_model_port import AIModelResult as AIModelResult\n",
                    "from .cache_port import CachePort as CachePort\n",
                    "from .dictionary_port import DictionaryPort as DictionaryPort\n",
                    "from .embed_model_port import EmbedModelPort as EmbedModelPort\n",
                    "from .ner_model_port import NERModelPort as NERModelPort\n",
                ),
            ),
            (
                "pkg/ai_model_port.py",
                "class AIModelPort: ...\nclass AIModelResult: ...\n",
            ),
            ("pkg/cache_port.py", "class CachePort: ...\n"),
            ("pkg/dictionary_port.py", "class DictionaryPort: ...\n"),
            ("pkg/embed_model_port.py", "class EmbedModelPort: ...\n"),
            ("pkg/ner_model_port.py", "class NERModelPort: ...\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/ner_model_port.py"),
            SystemPath::new("pkg/ner_model.py"),
        );

        assert_eq!(edits.len(), 1);

        let init = system_path_to_file(&db, "pkg/__init__.py").unwrap();
        let result = apply_edits(&db, &edits, init);
        let expected = concat!(
            "from .ai_model_port import AIModelPort as AIModelPort\n",
            "from .ai_model_port import AIModelResult as AIModelResult\n",
            "from .cache_port import CachePort as CachePort\n",
            "from .dictionary_port import DictionaryPort as DictionaryPort\n",
            "from .embed_model_port import EmbedModelPort as EmbedModelPort\n",
            "from .ner_model import NERModelPort as NERModelPort\n",
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn rename_from_parent_import_submodule() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/old_sub.py", "x = 1\n"),
            ("consumer.py", "from pkg import old_sub\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/old_sub.py"),
            SystemPath::new("pkg/new_sub.py"),
        );

        assert_eq!(edits.len(), 1);

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from pkg import new_sub\n");
    }

    #[test]
    fn rename_module_usage_sites() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\ndef hello(): ...\n"),
            (
                "consumer.py",
                "import old_module\n\nprint(old_module.x)\nold_module.hello()\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(
            result,
            "import new_module\n\nprint(new_module.x)\nnew_module.hello()\n"
        );
    }

    #[test]
    fn rename_dotted_import() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/old_sub.py", "x = 1\n"),
            (
                "consumer.py",
                "import pkg.old_sub\n\nprint(pkg.old_sub.x)\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/old_sub.py"),
            SystemPath::new("pkg/new_sub.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import pkg.new_sub\n\nprint(pkg.new_sub.x)\n");
    }

    #[test]
    fn rename_cross_directory() {
        let db = create_test_db(&[
            ("/old_package/__init__.py", ""),
            ("/old_package/old_module.py", "x = 1\n"),
            ("/new_package/__init__.py", ""),
            (
                "/consumer.py",
                "import old_package.old_module\n\nprint(old_package.old_module.x)\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old_package/old_module.py"),
            SystemPath::new("/new_package/new_module.py"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(
            result,
            "import new_package.new_module\n\nprint(new_package.new_module.x)\n"
        );
    }

    #[test]
    fn rename_cross_directory_from_import() {
        let db = create_test_db(&[
            ("/old_package/__init__.py", ""),
            ("/old_package/old_module.py", "x = 1\n"),
            ("/new_package/__init__.py", ""),
            ("/consumer.py", "from old_package.old_module import x\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old_package/old_module.py"),
            SystemPath::new("/new_package/new_module.py"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from new_package.new_module import x\n");
    }

    #[test]
    fn rename_cross_directory_standalone_import() {
        let db = create_test_db(&[
            ("/old_package/__init__.py", ""),
            ("/old_package/old_module.py", "x = 1\n"),
            ("/new_package/__init__.py", ""),
            (
                "/consumer.py",
                "from old_package import old_module\nprint(old_module.x)\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old_package/old_module.py"),
            SystemPath::new("/new_package/new_module.py"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(
            result,
            "from new_package import new_module\nprint(new_module.x)\n"
        );
    }

    #[test]
    fn rename_cross_directory_rewrites_relative_import_in_moved_file() {
        let db = create_test_db(&[
            ("/old_package/__init__.py", ""),
            ("/old_package/helper.py", "x = 1\n"),
            ("/old_package/moved.py", "from . import helper\n"),
            ("/new_package/__init__.py", ""),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old_package/moved.py"),
            SystemPath::new("/new_package/moved.py"),
        );

        let moved = system_path_to_file(&db, "/old_package/moved.py").unwrap();
        let result = apply_edits(&db, &edits, moved);
        assert_eq!(result, "from old_package import helper\n");
    }

    #[test]
    fn rename_cross_directory_import_with_sibling_is_conservative() {
        let db = create_test_db(&[
            ("/old_package/__init__.py", ""),
            ("/old_package/old_module.py", ""),
            ("/old_package/other.py", ""),
            ("/new_package/__init__.py", ""),
            (
                "/consumer.py",
                "from old_package import old_module, other\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old_package/old_module.py"),
            SystemPath::new("/new_package/new_module.py"),
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_top_level_module_into_package() {
        let db = create_test_db(&[
            ("/old.py", "x = 1\n"),
            ("/pkg/__init__.py", ""),
            ("/consumer.py", "import old\nprint(old.x)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old.py"),
            SystemPath::new("/pkg/new.py"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import pkg.new\nprint(pkg.new.x)\n");
    }

    #[test]
    fn rename_package_submodule_to_top_level() {
        let db = create_test_db(&[
            ("/pkg/__init__.py", ""),
            ("/pkg/old.py", "x = 1\n"),
            ("/consumer.py", "from pkg import old\nprint(old.x)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/pkg/old.py"),
            SystemPath::new("/new.py"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import new\nprint(new.x)\n");
    }

    #[test]
    fn rename_preserves_relative_import() {
        let db = create_test_db(&[
            ("/root/__init__.py", ""),
            ("/root/pkg/__init__.py", ""),
            ("/root/pkg/old.py", "x = 1\n"),
            ("/root/consumer.py", "from .pkg.old import x\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("/root/pkg/old.py"),
            SystemPath::new("/root/pkg/new.py"),
        );

        let consumer = system_path_to_file(&db, "/root/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from .pkg.new import x\n");
    }

    #[test]
    fn rename_updates_absolute_self_import_in_moved_package() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", "from old_pkg.sub import x\n"),
            ("/old_pkg/sub.py", "x = 1\n"),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/old_pkg"),
            SystemPath::new("/new_pkg"),
        );

        let init = system_path_to_file(&db, "/old_pkg/__init__.py").unwrap();
        let result = apply_edits(&db, &edits, init);
        assert_eq!(result, "from new_pkg.sub import x\n");
    }

    #[test]
    fn rename_updates_references_in_excluded_anchor_file() {
        let mut db = create_test_db(&[
            ("/old_module.py", "import old_module\n"),
            ("/consumer.py", ""),
        ]);
        let project = db.project();
        project.set_included_paths(&mut db, vec!["/consumer.py".into()]);
        let old_module = system_path_to_file(&db, "/old_module.py").unwrap();
        assert!(!project.files(&db).contains(&old_module));

        let edits = will_rename_file(
            &db,
            SystemPath::new("/old_module.py"),
            SystemPath::new("/new_module.py"),
        );

        let result = apply_edits(&db, &edits, old_module);
        assert_eq!(result, "import new_module\n");
    }

    #[test]
    fn rename_init_file_returns_no_edits() {
        let db = create_test_db(&[
            ("pkg/__init__.py", "x = 1\n"),
            ("consumer.py", "import pkg\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/__init__.py"),
            SystemPath::new("pkg/new.py"),
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_shadowed_module_not_rewritten() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/foo.py", "x = 1\n"),
            (
                "consumer.py",
                "from pkg import foo\n\ndef f(pkg):\n    return pkg.foo\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/foo.py"),
            SystemPath::new("pkg/bar.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        // `from pkg import foo` → `from pkg import bar`, but `pkg.foo` inside
        // `f(pkg)` should NOT be rewritten because `pkg` is a parameter.
        assert_eq!(
            result,
            "from pkg import bar\n\ndef f(pkg):\n    return pkg.foo\n"
        );
    }

    #[test]
    fn rename_does_not_rewrite_attribute_on_aliased_module() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/foo.py", "x = 1\n"),
            ("other.py", "foo = 1\n"),
            ("consumer.py", "import other as pkg\nprint(pkg.foo)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/foo.py"),
            SystemPath::new("pkg/bar.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import other as pkg\nprint(pkg.foo)\n");
    }

    #[test]
    fn rename_does_not_rewrite_package_attribute() {
        let db = create_test_db(&[
            ("pkg/__init__.py", "foo = 1\n"),
            ("pkg/foo.py", "x = 1\n"),
            ("consumer.py", "import pkg\nprint(pkg.foo)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/foo.py"),
            SystemPath::new("pkg/bar.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import pkg\nprint(pkg.foo)\n");
    }

    #[test]
    fn rename_does_not_rewrite_module_reexport_attribute() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("pkg/__init__.py", "import old_module as old_module\n"),
            ("consumer.py", "import pkg\nprint(pkg.old_module.x)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.py"),
        );

        let init = system_path_to_file(&db, "pkg/__init__.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, init),
            "import new_module as old_module\n"
        );
        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        assert_eq!(
            apply_edits(&db, &edits, consumer),
            "import pkg\nprint(pkg.old_module.x)\n"
        );
    }

    #[test]
    fn rename_module_access_through_aliased_parent() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/old_sub.py", "x = 1\n"),
            (
                "consumer.py",
                "import pkg as p\nimport pkg.old_sub\nprint(p.old_sub.x)\n",
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/old_sub.py"),
            SystemPath::new("pkg/new_sub.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(
            result,
            "import pkg as p\nimport pkg.new_sub\nprint(p.new_sub.x)\n"
        );
    }

    #[test]
    fn rename_keeps_explicit_alias_uses_stable_across_scopes() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/old_sub.py", "x = 1\n"),
            (
                "consumer.py",
                concat!(
                    "def outer():\n",
                    "    return old_sub.x\n",
                    "\n",
                    "import pkg.old_sub as old_sub\n",
                    "\n",
                    "def inner():\n",
                    "    from pkg import old_sub\n",
                    "    return old_sub.x\n",
                ),
            ),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/old_sub.py"),
            SystemPath::new("pkg/new_sub.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(
            result,
            concat!(
                "def outer():\n",
                "    return old_sub.x\n",
                "\n",
                "import pkg.new_sub as old_sub\n",
                "\n",
                "def inner():\n",
                "    from pkg import new_sub\n",
                "    return new_sub.x\n",
            )
        );
    }

    #[test]
    fn rename_formatted_from_import() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/old_sub.py", "x = 1\n"),
            ("consumer.py", "from pkg . old_sub import x\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("pkg/old_sub.py"),
            SystemPath::new("pkg/new_sub.py"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from pkg.new_sub import x\n");
    }

    #[test]
    fn rename_pyi_file() {
        let db = create_test_db(&[
            ("old_module.pyi", "x: int\n"),
            ("consumer.py", "import old_module\n\nprint(old_module.x)\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.pyi"),
            SystemPath::new("new_module.pyi"),
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import new_module\n\nprint(new_module.x)\n");
    }

    #[test]
    fn rename_py_and_pyi_batch_deduplicates_edits() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("old_module.pyi", "x: int\n"),
            ("consumer.py", "import old_module\nprint(old_module.x)\n"),
        ]);

        let edits = will_rename_paths(
            &db,
            &[
                PathRename::file("old_module.py".into(), "new_module.py".into()),
                PathRename::file("old_module.pyi".into(), "new_module.pyi".into()),
            ],
        );

        let consumer = system_path_to_file(&db, "consumer.py").unwrap();
        assert_eq!(edits.iter().filter(|edit| edit.file == consumer).count(), 2);
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import new_module\nprint(new_module.x)\n");
    }

    #[test]
    fn rename_batch_does_not_return_conflicting_edits() {
        let db = create_test_db(&[
            ("pkg/__init__.py", ""),
            ("pkg/a.py", ""),
            ("pkg/b.py", ""),
            ("x/__init__.py", ""),
            ("y/__init__.py", ""),
            ("consumer.py", "from pkg import a, b\n"),
        ]);

        let edits = will_rename_paths(
            &db,
            &[
                PathRename::file("pkg/a.py".into(), "x/a.py".into()),
                PathRename::file("pkg/b.py".into(), "y/b.py".into()),
            ],
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_batch_with_conflicting_module_destinations_returns_no_edits() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("old_module.pyi", "x: int\n"),
            ("other.py", "y = 1\n"),
            (
                "consumer.py",
                "import old_module\nimport other\nprint(old_module.x, other.y)\n",
            ),
        ]);

        let edits = will_rename_paths(
            &db,
            &[
                PathRename::file("old_module.py".into(), "first.py".into()),
                PathRename::file("old_module.pyi".into(), "second.pyi".into()),
                PathRename::file("other.py".into(), "new_other.py".into()),
            ],
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_directory_simple() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", ""),
            ("/old_pkg/sub.py", "x = 1\n"),
            ("/consumer.py", "import old_pkg\n\nprint(old_pkg.sub)\n"),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/old_pkg"),
            SystemPath::new("/new_pkg"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import new_pkg\n\nprint(new_pkg.sub)\n");
    }

    #[test]
    fn rename_directory_from_import() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", "x = 1\n"),
            ("/consumer.py", "from old_pkg import x\n"),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/old_pkg"),
            SystemPath::new("/new_pkg"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from new_pkg import x\n");
    }

    #[test]
    fn rename_directory_dotted_import() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", ""),
            ("/old_pkg/sub.py", "x = 1\n"),
            (
                "/consumer.py",
                "import old_pkg.sub\nfrom old_pkg.sub import x\n",
            ),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/old_pkg"),
            SystemPath::new("/new_pkg"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "import new_pkg.sub\nfrom new_pkg.sub import x\n");
    }

    #[test]
    fn rename_directory_no_init_skipped() {
        let db = create_test_db(&[
            ("/ns_pkg/sub.py", "x = 1\n"),
            ("/consumer.py", "from ns_pkg import sub\n"),
        ]);

        let edits =
            will_rename_directory(&db, SystemPath::new("/ns_pkg"), SystemPath::new("/new_ns"));

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_directory_nested_package() {
        let db = create_test_db(&[
            ("/parent/__init__.py", ""),
            ("/parent/old_child/__init__.py", "x = 1\n"),
            ("/parent/old_child/mod.py", "y = 2\n"),
            (
                "/consumer.py",
                "from parent.old_child import x\nimport parent.old_child.mod\n",
            ),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/parent/old_child"),
            SystemPath::new("/parent/new_child"),
        );

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(
            result,
            "from parent.new_child import x\nimport parent.new_child.mod\n"
        );
    }

    #[test]
    fn rename_directory_relative_imports_unchanged() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", ""),
            ("/old_pkg/a.py", "from . import b\n"),
            ("/old_pkg/b.py", "x = 1\n"),
            ("/consumer.py", "from old_pkg import a\n"),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/old_pkg"),
            SystemPath::new("/new_pkg"),
        );

        // Only consumer.py should be modified; files inside the package must not
        // appear in edits (relative imports are unaffected by package rename).
        for edit in &edits {
            let path = edit.file.path(&db);
            assert!(
                !path.as_str().contains("old_pkg"),
                "unexpected edit in package-internal file: {path}"
            );
        }

        let consumer = system_path_to_file(&db, "/consumer.py").unwrap();
        let result = apply_edits(&db, &edits, consumer);
        assert_eq!(result, "from new_pkg import a\n");
    }

    #[test]
    fn rename_directory_cross_package_rewrites_external_relative_import() {
        let db = create_test_db(&[
            ("/parent/__init__.py", ""),
            ("/parent/sibling.py", "x = 1\n"),
            ("/parent/old_pkg/__init__.py", ""),
            ("/parent/old_pkg/mod.py", "from ..sibling import x\n"),
            ("/other/__init__.py", ""),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/parent/old_pkg"),
            SystemPath::new("/other/new_pkg"),
        );

        let module = system_path_to_file(&db, "/parent/old_pkg/mod.py").unwrap();
        let result = apply_edits(&db, &edits, module);
        assert_eq!(result, "from parent.sibling import x\n");
    }

    #[test]
    fn rename_directory_invalid_identifier_returns_no_edits() {
        let db = create_test_db(&[
            ("/old_pkg/__init__.py", "x = 1\n"),
            ("/consumer.py", "from old_pkg import x\n"),
        ]);

        let edits = will_rename_directory(
            &db,
            SystemPath::new("/old_pkg"),
            SystemPath::new("/123-bad"),
        );

        assert!(edits.is_empty());
    }

    #[test]
    fn rename_to_non_python_file_returns_no_edits() {
        let db = create_test_db(&[
            ("old_module.py", "x = 1\n"),
            ("consumer.py", "import old_module\n"),
        ]);

        let edits = will_rename_file(
            &db,
            SystemPath::new("old_module.py"),
            SystemPath::new("new_module.txt"),
        );

        assert!(edits.is_empty());
    }

    fn will_rename_file(
        db: &dyn Db,
        old_path: &SystemPath,
        new_path: &SystemPath,
    ) -> Vec<FileRenameEdit> {
        will_rename_paths(
            db,
            &[PathRename::file(
                old_path.to_path_buf(),
                new_path.to_path_buf(),
            )],
        )
    }

    fn will_rename_directory(
        db: &dyn Db,
        old_dir: &SystemPath,
        new_dir: &SystemPath,
    ) -> Vec<FileRenameEdit> {
        will_rename_paths(
            db,
            &[PathRename::directory(
                old_dir.to_path_buf(),
                new_dir.to_path_buf(),
            )],
        )
    }

    fn create_test_db(files: &[(&str, &str)]) -> TestDb {
        let mut db = TestDb::new(ProjectMetadata::new(
            "test".into(),
            SystemPathBuf::from("/"),
        ));

        db.init_program_with_python_version(PythonVersion::latest_ty())
            .unwrap();

        for &(path, contents) in files {
            db.write_file(path, contents)
                .expect("write to memory file system to be successful");
        }

        db
    }

    fn apply_edits(db: &dyn Db, edits: &[FileRenameEdit], file: File) -> String {
        let source = source_text(db, file);
        let text = source.as_str().to_owned();

        let mut sorted_edits: Vec<_> = edits.iter().filter(|e| e.file == file).collect();
        sorted_edits.sort_by_key(|b| std::cmp::Reverse(b.range.start()));

        let mut result = text;
        for edit in sorted_edits {
            let start = usize::from(edit.range.start());
            let end = usize::from(edit.range.end());
            result.replace_range(start..end, &edit.new_text);
        }
        result
    }
}
