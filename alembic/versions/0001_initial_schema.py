"""Create NoteBot persistence schema.

Revision ID: 0001_initial_schema
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.config import settings

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("page_count", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "document_pages",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(256), primary_key=True),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(settings.embedding_dimensions)),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE")),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("conversation_id", sa.UUID(), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "message_sources",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("message_id", sa.UUID(), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_id", sa.String(256), sa.ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
    )
    op.create_table(
        "generated_assets",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("document_id", sa.UUID(), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("asset_type", sa.String(32), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Gemini's 3072-dimensional embeddings exceed pgvector's 2000-dimension
    # HNSW index limit. Cosine search remains available through pgvector and
    # can be indexed later with a dimension-reduction strategy.
    op.execute(
        "CREATE INDEX chunks_document_page_idx ON chunks "
        "(document_id, page_number, chunk_index)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_document_page_idx")
    op.drop_table("generated_assets")
    op.drop_table("message_sources")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("chunks")
    op.drop_table("document_pages")
    op.drop_table("documents")
    op.drop_table("users")
