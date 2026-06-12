#include "position.hpp"

#include <cassert>
#include <cctype>
#include <ostream>
#include <string_view>
#include "zobrist.hpp"

namespace chess {

namespace {

constexpr int color_index(Color color) {
    return static_cast<int>(color);
}

constexpr int piece_index(PieceType piece) {
    return static_cast<int>(piece);
}

char piece_char(Color color, PieceType piece) {
    constexpr std::array<char, 6> WhiteChars{'P', 'N', 'B', 'R', 'Q', 'K'};
    constexpr std::array<char, 6> BlackChars{'p', 'n', 'b', 'r', 'q', 'k'};

    const int index = piece_index(piece);
    return color == Color::White ? WhiteChars[index] : BlackChars[index];
}

char piece_on_square(const Position& position, Square square) {
    for (Color color : {Color::White, Color::Black}) {
        for (PieceType piece :
             {PieceType::Pawn, PieceType::Knight, PieceType::Bishop,
              PieceType::Rook, PieceType::Queen, PieceType::King}) {
            const Bitboard bb = position.pieces[color_index(color)][piece_index(piece)];
            if (bb & bit(square)) {
                return piece_char(color, piece);
            }
        }
    }

    return '.';
}

bool piece_from_char(char ch, Color& color, PieceType& piece) {
    switch (ch) {
        case 'P':
            color = Color::White;
            piece = PieceType::Pawn;
            return true;
        case 'N':
            color = Color::White;
            piece = PieceType::Knight;
            return true;
        case 'B':
            color = Color::White;
            piece = PieceType::Bishop;
            return true;
        case 'R':
            color = Color::White;
            piece = PieceType::Rook;
            return true;
        case 'Q':
            color = Color::White;
            piece = PieceType::Queen;
            return true;
        case 'K':
            color = Color::White;
            piece = PieceType::King;
            return true;
        case 'p':
            color = Color::Black;
            piece = PieceType::Pawn;
            return true;
        case 'n':
            color = Color::Black;
            piece = PieceType::Knight;
            return true;
        case 'b':
            color = Color::Black;
            piece = PieceType::Bishop;
            return true;
        case 'r':
            color = Color::Black;
            piece = PieceType::Rook;
            return true;
        case 'q':
            color = Color::Black;
            piece = PieceType::Queen;
            return true;
        case 'k':
            color = Color::Black;
            piece = PieceType::King;
            return true;
        default:
            return false;
    }
}

void skip_spaces(std::string_view text, std::size_t& index) {
    while (index < text.size() && text[index] == ' ') {
        ++index;
    }
}

bool parse_non_negative_int(std::string_view text, std::size_t& index, int& value) {
    if (index == text.size() || !std::isdigit(static_cast<unsigned char>(text[index]))) {
        return false;
    }

    int parsed = 0;
    while (index < text.size() && std::isdigit(static_cast<unsigned char>(text[index]))) {
        parsed = parsed * 10 + (text[index] - '0');
        ++index;
    }

    value = parsed;
    return true;
}


consteval std::array<std::array<Bitboard, 6>, 2> make_start_pieces() {
    std::array<std::array<Bitboard, 6>, 2> pieces{};

    auto set = [&](Color color, PieceType piece, Square square) {
        pieces[static_cast<int>(color)][static_cast<int>(piece)] |= bit(square);
    };

    for (int file = 0; file < 8; ++file) {
        set(Color::White, PieceType::Pawn, make_square(file, 1));
        set(Color::Black, PieceType::Pawn, make_square(file, 6));
    }

    set(Color::White, PieceType::Rook,   make_square(0, 0));
    set(Color::White, PieceType::Knight, make_square(1, 0));
    set(Color::White, PieceType::Bishop, make_square(2, 0));
    set(Color::White, PieceType::Queen,  make_square(3, 0));
    set(Color::White, PieceType::King,   make_square(4, 0));
    set(Color::White, PieceType::Bishop, make_square(5, 0));
    set(Color::White, PieceType::Knight, make_square(6, 0));
    set(Color::White, PieceType::Rook,   make_square(7, 0));

    set(Color::Black, PieceType::Rook,   make_square(0, 7));
    set(Color::Black, PieceType::Knight, make_square(1, 7));
    set(Color::Black, PieceType::Bishop, make_square(2, 7));
    set(Color::Black, PieceType::Queen,  make_square(3, 7));
    set(Color::Black, PieceType::King,   make_square(4, 7));
    set(Color::Black, PieceType::Bishop, make_square(5, 7));
    set(Color::Black, PieceType::Knight, make_square(6, 7));
    set(Color::Black, PieceType::Rook,   make_square(7, 7));

    return pieces;
}

constexpr auto StartPieces = make_start_pieces();


} // namespace

void Position::print(std::ostream& os) const {
    for (int rank = 7; rank >= 0; --rank) {
        for (int file = 0; file < 8; ++file) {
            const Square square = make_square(file, rank);
            os << piece_on_square(*this, square);

            if (file != 7) {
                os << ' ';
            }
        }
        os << '\n';
    }
}

Bitboard Position::occupancy(Color color) const {
    Bitboard ret = EmptyBB;
    const int color_idx = color_index(color);

    for (Bitboard bb : pieces[color_idx]) {
        ret |= bb;
    }

    return ret;
}

Bitboard Position::occupancy(Color color, PieceType piece) const {
    return pieces[static_cast<int>(color)][static_cast<int>(piece)];
}

Bitboard Position::occupancy() const {
    return occupancy(Color::White) | occupancy(Color::Black);
}

void Position::set_piece(Color color, PieceType piece, Square square) {
    assert(piece != PieceType::None);
    assert((occupancy() & bit(square)) == EmptyBB);

    pieces[color_index(color)][piece_index(piece)] |= bit(square);
    zobrist_key ^= zobrist::piece_key(color, piece, square);
}

void Position::clear_square(Square square) {
    const Bitboard clear_mask = ~bit(square);

    for (int colorIndex = 0; colorIndex < 2; colorIndex++) {
        for (int pieceIndex = 0; pieceIndex < 6; pieceIndex++) {
            Bitboard& bb = pieces[colorIndex][pieceIndex];
            Bitboard new_bb = bb & clear_mask;
            if (new_bb != bb) {
                bb = new_bb;
                zobrist_key ^= zobrist::piece_key(colorIndex, pieceIndex, static_cast<int>(square));
                return;
            }
        }
    }
}

Color Position::color_on_occupied(Square square) const {
    assert(!is_empty(square));

    const Bitboard square_mask = bit(square);
    if (occupancy(Color::White) & square_mask) {
        return Color::White;
    }

    assert(occupancy(Color::Black) & square_mask);
    return Color::Black;
}

PieceType Position::piece_type_on_occupied(Square square) const {
    assert(!is_empty(square));

    const Bitboard square_mask = bit(square);
    for (const auto& color_pieces : pieces) {
        for (std::size_t piece_index = 0; piece_index < color_pieces.size(); ++piece_index) {
            if (color_pieces[piece_index] & square_mask) {
                return static_cast<PieceType>(piece_index);
            }
        }
    }

    assert(false);
    return PieceType::None;
}

bool Position::is_empty(Square square) const {
    return (occupancy() & bit(square)) == EmptyBB;
}

void Position::clear() {
    for (auto &color_pieces : pieces) {
        for (auto& bb : color_pieces) {
            bb = EmptyBB;
        }
    }
    side_to_move = Color::White;
    white_can_castle_kingside = false;
    white_can_castle_queenside = false;
    black_can_castle_kingside = false;
    black_can_castle_queenside = false;
    en_passant_square = NoSquare;
    halfmove_clock = 0;
    fullmove_number = 1;
    zobrist_key = 0;
}

void Position::set_startpos() {
    pieces = StartPieces;
    side_to_move = Color::White;
    white_can_castle_kingside = true;
    white_can_castle_queenside = true;
    black_can_castle_kingside = true;
    black_can_castle_queenside = true;
    en_passant_square = NoSquare;
    halfmove_clock = 0;
    fullmove_number = 1;
    zobrist_key = zobrist::compute_hash(*this);
}

bool Position::set_fen(std::string_view fen) {
    Position tmp;
    tmp.clear();

    std::size_t i = 0;
    int rank = 7;
    int file = 0;

    while (i < fen.size() && fen[i] != ' ') {
        const char ch = fen[i];

        if (ch == '/') {
            if (file != 8 || rank == 0) {
                return false;
            }

            --rank;
            file = 0;
        } else if ('1' <= ch && ch <= '8') {
            file += ch - '0';
            if (file > 8) {
                return false;
            }
        } else {
            Color color = Color::White;
            PieceType piece = PieceType::None;
            if (!piece_from_char(ch, color, piece) || file >= 8) {
                return false;
            }

            tmp.set_piece(color, piece, make_square(file, rank));
            ++file;
        }

        ++i;
    }

    if (rank != 0 || file != 8) {
        return false;
    }

    if (i == fen.size() || fen[i] != ' ') {
        return false;
    }

    skip_spaces(fen, i);

    if (i == fen.size()) {
        return false;
    }

    if (fen[i] == 'w') {
        tmp.side_to_move = Color::White;
    } else if (fen[i] == 'b') {
        tmp.side_to_move = Color::Black;
    } else {
        return false;
    }

    ++i;
    if (i < fen.size() && fen[i] != ' ') {
        return false;
    }

    skip_spaces(fen, i);

    if (i < fen.size()) {
        if (fen[i] == '-') {
            ++i;
            if (i < fen.size() && fen[i] != ' ') {
                return false;
            }
        } else {
            bool seen_white_kingside = false;
            bool seen_white_queenside = false;
            bool seen_black_kingside = false;
            bool seen_black_queenside = false;

            while (i < fen.size() && fen[i] != ' ') {
                switch (fen[i]) {
                    case 'K':
                        if (seen_white_kingside) {
                            return false;
                        }
                        seen_white_kingside = true;
                        tmp.white_can_castle_kingside = true;
                        break;
                    case 'Q':
                        if (seen_white_queenside) {
                            return false;
                        }
                        seen_white_queenside = true;
                        tmp.white_can_castle_queenside = true;
                        break;
                    case 'k':
                        if (seen_black_kingside) {
                            return false;
                        }
                        seen_black_kingside = true;
                        tmp.black_can_castle_kingside = true;
                        break;
                    case 'q':
                        if (seen_black_queenside) {
                            return false;
                        }
                        seen_black_queenside = true;
                        tmp.black_can_castle_queenside = true;
                        break;
                    default:
                        return false;
                }
                ++i;
            }
        }
    }

    skip_spaces(fen, i);

    if (i < fen.size()) {
        if (fen[i] == '-') {
            tmp.en_passant_square = NoSquare;
            ++i;
        } else {
            if (i + 1 >= fen.size()) {
                return false;
            }

            const char file_ch = fen[i];
            const char rank_ch = fen[i + 1];
            if (file_ch < 'a' || file_ch > 'h' || rank_ch < '1' || rank_ch > '8') {
                return false;
            }

            tmp.en_passant_square = make_square(file_ch - 'a', rank_ch - '1');
            i += 2;
        }

        if (i < fen.size() && fen[i] != ' ') {
            return false;
        }
    }

    skip_spaces(fen, i);

    if (i < fen.size()) {
        if (!parse_non_negative_int(fen, i, tmp.halfmove_clock)) {
            return false;
        }

        if (i < fen.size() && fen[i] != ' ') {
            return false;
        }
    }

    skip_spaces(fen, i);

    if (i < fen.size()) {
        if (!parse_non_negative_int(fen, i, tmp.fullmove_number) || tmp.fullmove_number <= 0) {
            return false;
        }

        skip_spaces(fen, i);
        if (i != fen.size()) {
            return false;
        }
    }

    tmp.zobrist_key = zobrist::compute_hash(tmp);
    *this = tmp;
    return true;
}

} 
