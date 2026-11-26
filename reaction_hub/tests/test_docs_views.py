from __future__ import annotations


def test_docs_index_lists_markdown_files(client):
    resp = client.get("/docs/")

    assert resp.status_code == 200
    content = resp.content.decode("utf-8", errors="replace")
    assert "README.md" in content
    assert "architecture.md" in content


def test_docs_detail_displays_markdown_content(client):
    resp = client.get("/docs/README.md")

    assert resp.status_code == 200
    content = resp.content.decode("utf-8", errors="replace")
    assert "Reaction Hub" in content


def test_docs_detail_blocks_path_traversal(client):
    resp = client.get("/docs/../README.md")

    assert resp.status_code == 404


def test_docs_detail_rejects_non_markdown_files(client):
    resp = client.get("/docs/examples/slurm_submission_template.sh.tmpl")

    assert resp.status_code == 404
