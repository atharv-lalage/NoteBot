"""Remove upload-version fields from logical documents."""

from alembic import op
import sqlalchemy as sa


revision = "0005_remove_legacy_docs"
down_revision = "0004_conversation_message_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Upload metadata was copied to document_versions by 0003. The documents
    # table now represents only the logical document identity.
    op.drop_constraint("documents_content_hash_key", "documents", type_="unique")
    for column in (
        "content_hash",
        "size_bytes",
        "page_count",
        "status",
        "error",
        "version_number",
        "version_group_id",
        "is_current",
    ):
        op.drop_column("documents", column)


def downgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("is_current", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("version_group_id", sa.UUID(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("version_number", sa.Integer(), nullable=True),
    )
    op.add_column("documents", sa.Column("error", sa.Text(), nullable=True))
    op.add_column(
        "documents",
        sa.Column("status", sa.String(32), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("page_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("size_bytes", sa.Integer(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(64), nullable=True),
    )
    op.execute(
        """
        UPDATE documents AS d
        SET version_group_id = d.id,
            version_number = v.version_number,
            content_hash = v.content_hash,
            size_bytes = v.size_bytes,
            page_count = v.page_count,
            status = v.status,
            error = v.error,
            is_current = v.is_current
        FROM document_versions AS v
        WHERE v.document_id = d.id AND v.is_current
        """
    )
    op.alter_column("documents", "version_group_id", nullable=False)
    op.alter_column("documents", "version_number", nullable=False)
    op.alter_column("documents", "content_hash", nullable=False)
    op.alter_column("documents", "size_bytes", nullable=False)
    op.alter_column("documents", "page_count", nullable=False)
    op.alter_column("documents", "status", nullable=False)
    op.alter_column("documents", "is_current", nullable=False)
    op.create_unique_constraint(
        "documents_content_hash_key",
        "documents",
        ["content_hash"],
    )
