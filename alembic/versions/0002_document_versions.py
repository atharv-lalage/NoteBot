"""Add immutable document version metadata and source snapshots."""

from alembic import op
import sqlalchemy as sa

revision = "0002_document_versions"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "documents",
        sa.Column("version_group_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.execute("UPDATE documents SET version_group_id = id WHERE version_group_id IS NULL")
    op.alter_column("documents", "version_group_id", nullable=False)
    op.add_column(
        "message_sources",
        sa.Column("document_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "message_sources",
        sa.Column("document_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "message_sources",
        sa.Column("page_number", sa.Integer(), nullable=True),
    )
    op.execute(
        """
        UPDATE message_sources AS ms
        SET document_id = c.document_id,
            document_version = d.version_number,
            page_number = c.page_number
        FROM chunks AS c
        JOIN documents AS d ON d.id = c.document_id
        WHERE ms.chunk_id = c.id
        """
    )
    op.alter_column("message_sources", "document_id", nullable=False)
    op.alter_column("message_sources", "document_version", nullable=False)
    op.alter_column("message_sources", "page_number", nullable=False)
    op.create_foreign_key(
        "message_sources_document_id_fkey",
        "message_sources",
        "documents",
        ["document_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index(
        "documents_version_group_current_idx",
        "documents",
        ["version_group_id", "is_current"],
    )


def downgrade() -> None:
    op.drop_index("documents_version_group_current_idx", table_name="documents")
    op.drop_constraint(
        "message_sources_document_id_fkey", "message_sources", type_="foreignkey"
    )
    op.drop_column("message_sources", "page_number")
    op.drop_column("message_sources", "document_version")
    op.drop_column("message_sources", "document_id")
    op.drop_column("documents", "is_current")
    op.drop_column("documents", "version_group_id")
    op.drop_column("documents", "version_number")
