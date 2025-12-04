# This file makes the 'commercial' directory a Python package.
# We export the MiniMax provider so it can be imported from this package.

from .minimax import MiniMaxProvider

__all__ = ["MiniMaxProvider"]