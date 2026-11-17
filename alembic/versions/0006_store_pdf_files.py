"""Store original PDF bytes for version previews.

Revision ID: 0006_store_pdf_files
Revises: 0005_remove_legacy_docs
"""

from alembic import op
import sqlalchemy as sa


revision = "0006_store_pdf_files"
down_revision = "0005_remove_legacy_docs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("pdf_data", sa.LargeBinary(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_versions", "pdf_data")
