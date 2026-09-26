"""init

Revision ID: a076a9ae2870
Revises: 
Create Date: 2026-09-27 01:38:11.291211

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a076a9ae2870'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'simulation_runs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('scenario', sa.String(), nullable=False),
        sa.Column('policy', sa.String(), nullable=False),
        sa.Column('seed', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('CREATED', 'RUNNING', 'COMPLETED', 'FAILED', 'TERMINATED', name='simulationstatus'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_reward', sa.Float(), nullable=True),
        sa.Column('survivors_rescued', sa.Integer(), nullable=True),
        sa.Column('evacuation_rate', sa.Float(), nullable=True),
        sa.Column('episode_length', sa.Integer(), nullable=True),
        sa.Column('survivors_remaining', sa.Integer(), nullable=True),
        sa.Column('casualties', sa.Integer(), nullable=True),
        sa.Column('inference_time', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('simulation_runs')
    op.execute('DROP TYPE simulationstatus;')
