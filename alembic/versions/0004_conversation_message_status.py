"""Add conversation titles and message lifecycle status."""

from alembic import op
import sqlalchemy as sa


revision = "0004_conversation_message_status"
down_revision = "0003_split_document_versions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "title",
            sa.String(512),
            nullable=False,
            server_default="New conversation",
        ),
    )
    op.add_column(
        "messages",
        sa.Column("status", sa.String(32), nullable=False, server_default="completed"),
    )
    op.add_column(
        "messages",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        "UPDATE messages SET completed_at = created_at WHERE status = 'completed'"
    )


def downgrade() -> None:
    op.drop_column("messages", "completed_at")
    op.drop_column("messages", "status")
    op.drop_column("conversations", "title")
