"""Connect4 Monte Carlo Tree Search module."""

from enum import IntEnum

import numpy as np


class StepResult(IntEnum):
    NOTHING = 0
    TIE = 1
    WIN = 2


class GameBoard:
    """Connect4 game board class."""

    def __init__(self, width: int, height: int, n: int):
        assert width >= n or height >= height, "Impossible to win with this combination of width, height and n"
        self.turn = 1
        self.board = np.zeros(shape=(height, width), dtype=np.float32)
        self.width = width
        self.height = height
        self.n_to_align = n
        self.n_items_in_column = np.zeros(width, dtype=np.int32)

        self.str_row = "+" + "-" * (self.width * 4 - 1) + "+"
        self.numbers = "|" + " ".join([f" {i} " for i in range(self.width)]) + "|"

    def valid_moves(self):
        """Get list of valid moves (i.e. not full columns)."""
        return self.n_items_in_column < self.height

    def clear(self):
        self.board = np.zeros(shape=(self.height, self.width), dtype=np.float32)
        self.n_items_in_column = np.zeros(self.width, dtype=np.int32)
        self.turn = 0

    def show(self) -> None:
        """Print out game board on console."""
        print(self.str_row)
        for j in range(self.height - 1, -1, -1):
            for i in range(self.width):
                match self.board[j, i]:
                    case 1:
                        print("| X", end=" ")
                    case -1:
                        print("| O", end=" ")
                    case _:
                        print("|  ", end=" ")
            print("|")
        print(self.str_row)
        print(self.numbers)
        print(self.str_row)

    def check_win(self, move_played: tuple[int, int]) -> bool:
        if self.check_rows(move_played):
            return True
        if self.check_cols(move_played):
            return True
        if self.check_diags(move_played):
            return True
        return False

    def check_tie(self) -> bool:
        """
        Check whether the game is a tie (i.e. the board is full).

        Note that it does not check for a win, so it should be called after check_win.
        """
        # If the last row is full, the game is a tie
        return bool(np.all(self.board[-1] != 0))

    def check_rows(self, move_played: tuple[int, int]) -> bool:
        row, col = move_played
        start_index = max(0, col - self.n_to_align + 1)
        end_index = min(self.width - self.n_to_align, col) + 1
        for start in range(start_index, end_index):
            slice = self.board[row, start : start + self.n_to_align]
            if np.all(slice == self.turn):
                return True
        return False

    def check_cols(self, move_played: tuple[int, int]) -> bool:
        row, col = move_played
        start_index = max(0, row - self.n_to_align + 1)
        end_index = min(self.height - self.n_to_align, row) + 1
        for start in range(start_index, end_index):
            slice = self.board[start : start + self.n_to_align, col]
            if np.all(slice == self.turn):
                return True
        return False

    def check_diags(self, move_played: tuple[int, int]) -> bool:
        row, col = move_played
        # count the adjacent items in the / diagonal
        n_adjacent = 0
        # Top right
        row_i, col_i = row + 1, col + 1
        while row_i < self.height and col_i < self.width and self.board[row_i, col_i] == self.turn:
            n_adjacent += 1
            row_i += 1
            col_i += 1
        # Bottom left
        row_i, col_i = row - 1, col - 1
        while row_i >= 0 and col_i >= 0 and self.board[row_i, col_i] == self.turn:
            n_adjacent += 1
            row_i -= 1
            col_i -= 1
        if n_adjacent >= self.n_to_align - 1:
            return True

        # Count adjacent items in the \ diagonal
        n_adjacent = 0
        # Top left
        row_i, col_i = row + 1, col - 1
        while row_i < self.height and col_i >= 0 and self.board[row_i, col_i] == self.turn:
            n_adjacent += 1
            row_i += 1
            col_i -= 1
        # Bottom right
        row_i, col_i = row - 1, col + 1
        while row_i >= 0 and col_i < self.width and self.board[row_i, col_i] == self.turn:
            n_adjacent += 1
            row_i -= 1
            col_i += 1

        return n_adjacent >= self.n_to_align - 1

    def play(self, column: int) -> StepResult:
        """Apply move to board.

        Args:
            column (int): Selected column index (between 0 and the number of cols - 1).

        Returns:
            bool: whether the player has won.
        """
        row_index = self.n_items_in_column[column]
        if row_index >= self.height:
            raise ValueError(f"Column {column} is full, use `valid_moves` to check valid moves.")
        self.n_items_in_column[column] += 1
        self.board[row_index, column] = self.turn
        if self.check_win((row_index, column)):
            result = StepResult.WIN
        elif self.check_tie():
            result = StepResult.TIE
        else:
            result = StepResult.NOTHING
        self.switch_turn()
        return result

    def switch_turn(self) -> None:
        """Switch turn between players."""
        self.turn = -self.turn


def test_win():
    board = GameBoard(4, 1, 2)
    assert board.play(0) == StepResult.NOTHING
    assert board.play(2) == StepResult.NOTHING
    assert board.play(1) == StepResult.WIN


def test_tie():
    board = GameBoard(4, 1, 2)
    assert board.play(0) == StepResult.NOTHING
    assert board.play(1) == StepResult.NOTHING
    assert board.play(2) == StepResult.NOTHING
    assert board.play(3) == StepResult.TIE


def test_win_diag():
    board = GameBoard(2, 2, 2)
    assert board.play(0) == StepResult.NOTHING
    assert board.play(1) == StepResult.NOTHING
    assert board.play(1) == StepResult.WIN

    board.clear()
    assert board.play(1) == StepResult.NOTHING
    assert board.play(1) == StepResult.NOTHING
    assert board.play(0) == StepResult.WIN


if __name__ == "__main__":
    test_win()
    test_tie()
    test_win_diag()
    print("All tests passed!")
