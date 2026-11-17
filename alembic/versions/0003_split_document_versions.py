"""Separate logical documents from uploaded document versions."""

from alembic import op
import sqlalchemy as sa


revision = "0003_split_document_versions"
down_revision = "0002_document_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("name", sa.String(512), nullable=True),
    )
    op.execute("UPDATE documents SET name = filename WHERE name IS NULL")
    op.alter_column("documents", "name", nullable=False)

    op.create_table(
        "document_versions",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "document_id",
            sa.UUID(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("document_id", "version_number", name="document_versions_number_key"),
    )
    op.execute(
        """
        INSERT INTO document_versions
            (id, document_id, version_number, filename, content_hash, size_bytes,
             page_count, status, error, is_current, created_at)
        SELECT id, id, version_number, filename, content_hash, size_bytes,
               page_count, status, error, is_current, created_at
        FROM documents
        """
    )

    for table in ("document_pages", "chunks", "conversations", "generated_assets"):
        op.add_column(table, sa.Column("document_version_id", sa.UUID(), nullable=True))

    op.execute(
        "UPDATE document_pages SET document_version_id = document_id"
    )
    op.execute("UPDATE chunks SET document_version_id = document_id")
    op.execute("UPDATE conversations SET document_version_id = document_id")
    op.execute("UPDATE generated_assets SET document_version_id = document_id")

    for table in ("document_pages", "chunks", "conversations", "generated_assets"):
        op.alter_column(table, "document_version_id", nullable=False)
        op.create_foreign_key(
            f"{table}_document_version_id_fkey",
            table,
            "document_versions",
            ["document_version_id"],
            ["id"],
            ondelete="CASCADE",
        )

    op.add_column(
        "message_sources",
        sa.Column("document_version_id", sa.UUID(), nullable=True),
    )
    op.execute(
        """
        UPDATE message_sources AS ms
        SET document_version_id = c.document_version_id
        FROM chunks AS c
        WHERE ms.chunk_id = c.id
        """
    )
    op.alter_column("message_sources", "document_version_id", nullable=False)
    op.create_foreign_key(
        "message_sources_document_version_id_fkey",
        "message_sources",
        "document_versions",
        ["document_version_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "document_versions_current_idx",
        "document_versions",
        ["document_id", "is_current"],
    )
    op.create_index(
        "document_versions_one_current_idx",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("is_current"),
    )


def downgrade() -> None:
    op.drop_index("document_versions_one_current_idx", table_name="document_versions")
    op.drop_index("document_versions_current_idx", table_name="document_versions")
    op.drop_constraint(
        "message_sources_document_version_id_fkey",
        "message_sources",
        type_="foreignkey",
    )
    op.drop_column("message_sources", "document_version_id")
    for table in ("generated_assets", "conversations", "chunks", "document_pages"):
        op.drop_constraint(
            f"{table}_document_version_id_fkey",
            table,
            type_="foreignkey",
        )
        op.drop_column(table, "document_version_id")
    op.drop_table("document_versions")
    op.drop_column("documents", "name")
