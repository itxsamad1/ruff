use std::collections::HashMap;

use lsp_types::{
    DocumentChange, OptionalVersionedTextDocumentIdentifier, RenameFilesParams, TextDocumentEdit,
    TextDocumentIdentifier, TextEdit, Uri, WillRenameFilesRequest, WorkspaceEdit,
};
use ruff_db::system::SystemPathBuf;
use ty_ide::{PathRename, will_rename_paths};

use crate::document::ToRangeExt;
use crate::server::api::traits::{
    BackgroundRequestHandler, RequestHandler, RetriableRequestHandler,
};
use crate::session::SessionSnapshot;
use crate::session::client::Client;
use crate::system::file_to_uri;

pub(crate) struct WillRenameFilesHandler;

impl RequestHandler for WillRenameFilesHandler {
    type RequestType = WillRenameFilesRequest;
}

impl BackgroundRequestHandler for WillRenameFilesHandler {
    fn run(
        snapshot: &SessionSnapshot,
        _client: &Client,
        params: RenameFilesParams,
    ) -> crate::server::Result<Option<WorkspaceEdit>> {
        let encoding = snapshot.position_encoding();
        let mut all_changes: HashMap<Uri, Vec<TextEdit>> = HashMap::new();
        let mut renames_by_project = vec![Vec::new(); snapshot.projects().len()];

        for file_rename in &params.files {
            let Ok(old_uri) = Uri::parse(&file_rename.old_uri) else {
                continue;
            };
            let Ok(new_uri) = Uri::parse(&file_rename.new_uri) else {
                continue;
            };

            let Ok(old_std_path) = old_uri.to_file_path() else {
                continue;
            };
            let Ok(new_std_path) = new_uri.to_file_path() else {
                continue;
            };

            let Ok(old_path) = SystemPathBuf::from_path_buf(old_std_path) else {
                continue;
            };
            let Ok(new_path) = SystemPathBuf::from_path_buf(new_std_path) else {
                continue;
            };

            let has_python_extension = matches!(old_path.extension(), Some("py" | "pyi"));
            let Some(project_index) = snapshot.project_index_for_path(&old_path) else {
                continue;
            };
            let rename = if has_python_extension {
                PathRename::file(old_path, new_path)
            } else {
                PathRename::directory(old_path, new_path)
            };
            renames_by_project[project_index].push(rename);
        }

        for (db, renames) in snapshot.projects().iter().zip(renames_by_project) {
            for edit in will_rename_paths(db, &renames) {
                let (file, range, new_text) = edit.into_parts();
                let Some(uri) = file_to_uri(db, file) else {
                    continue;
                };

                let Some(lsp_range) = range.to_lsp_range(db, file, encoding) else {
                    continue;
                };

                all_changes.entry(uri).or_default().push(TextEdit {
                    range: lsp_range.local_range(),
                    new_text,
                });
            }
        }

        if all_changes
            .values_mut()
            .any(|edits| !normalize_text_edits(edits))
        {
            tracing::warn!("Skipping file-rename edits because the batch produced conflicts");
            return Ok(None);
        }

        if all_changes.is_empty() {
            Ok(None)
        } else if snapshot
            .resolved_client_capabilities()
            .supports_workspace_edit_document_changes()
        {
            let document_changes = all_changes
                .into_iter()
                .map(|(uri, edits)| {
                    DocumentChange::TextDocumentEdit(TextDocumentEdit {
                        text_document: OptionalVersionedTextDocumentIdentifier {
                            text_document_identifier: TextDocumentIdentifier { uri },
                            version: None,
                        },
                        edits: edits.into_iter().map(lsp_types::Edit::TextEdit).collect(),
                    })
                })
                .collect();

            Ok(Some(WorkspaceEdit {
                document_changes: Some(document_changes),
                ..Default::default()
            }))
        } else {
            Ok(Some(WorkspaceEdit {
                changes: Some(all_changes),
                ..Default::default()
            }))
        }
    }
}

impl RetriableRequestHandler for WillRenameFilesHandler {}

fn normalize_text_edits(edits: &mut Vec<TextEdit>) -> bool {
    edits.sort_by(|left, right| {
        left.range
            .start
            .cmp(&right.range.start)
            .then_with(|| left.range.end.cmp(&right.range.end))
            .then_with(|| left.new_text.cmp(&right.new_text))
    });
    edits.dedup();
    !edits.windows(2).any(|edits| {
        edits[0].range.start == edits[1].range.start || edits[0].range.end > edits[1].range.start
    })
}

#[cfg(test)]
mod tests {
    use lsp_types::{Position, Range};

    use super::*;

    #[test]
    fn normalizes_text_edits_and_rejects_conflicts() {
        let edit = TextEdit {
            range: Range::new(Position::new(0, 1), Position::new(0, 3)),
            new_text: "new".to_string(),
        };
        let mut duplicates = vec![edit.clone(), edit.clone()];
        assert!(normalize_text_edits(&mut duplicates));
        assert_eq!(duplicates.as_slice(), std::slice::from_ref(&edit));

        let mut conflicts = vec![
            edit,
            TextEdit {
                range: Range::new(Position::new(0, 2), Position::new(0, 4)),
                new_text: "other".to_string(),
            },
        ];
        assert!(!normalize_text_edits(&mut conflicts));
    }
}
