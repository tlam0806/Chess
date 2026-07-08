#include "king_safety_experimental.hpp"

#include "attacks.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cstdlib>
namespace chess::experimental {

namespace {

constexpr int abs_int(int value) {
    return value < 0 ? -value : value;
}

constexpr bool squares_aligned(Square a, Square b) {
    const int af = file_of(a);
    const int ar = rank_of(a);
    const int bf = file_of(b);
    const int br = rank_of(b);
    return af == bf
        || ar == br
        || abs_int(af - bf) == abs_int(ar - br);
}

constexpr Bitboard make_ray_to_mask(Square from, Square to) {
    if (from == to || !squares_aligned(from, to)) {
        return EmptyBB;
    }

    const int from_file = file_of(from);
    const int from_rank = rank_of(from);
    const int to_file = file_of(to);
    const int to_rank = rank_of(to);
    const int file_delta = (to_file > from_file) - (to_file < from_file);
    const int rank_delta = (to_rank > from_rank) - (to_rank < from_rank);

    Bitboard mask = EmptyBB;
    int file = from_file + file_delta;
    int rank = from_rank + rank_delta;
    while (file != to_file || rank != to_rank) {
        mask |= bit(make_square(file, rank));
        file += file_delta;
        rank += rank_delta;
    }
    return mask;
}

consteval std::array<Bitboard, BoardSize * BoardSize> make_ray_to_masks() {
    std::array<Bitboard, BoardSize * BoardSize> masks{};
    for (Square from = 0; from < BoardSize; ++from) {
        for (Square to = 0; to < BoardSize; ++to) {
            masks[static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(to)] =
                make_ray_to_mask(from, to);
        }
    }
    return masks;
}

constexpr auto RayToMasks = make_ray_to_masks();

constexpr Bitboard make_ray_to_edge_mask(Square from, Square through) {
    if (from == through || !squares_aligned(from, through)) {
        return EmptyBB;
    }

    const int from_file = file_of(from);
    const int from_rank = rank_of(from);
    const int through_file = file_of(through);
    const int through_rank = rank_of(through);
    const int file_delta = (through_file > from_file) - (through_file < from_file);
    const int rank_delta = (through_rank > from_rank) - (through_rank < from_rank);

    Bitboard mask = EmptyBB;
    int file = from_file + file_delta;
    int rank = from_rank + rank_delta;
    while (is_valid_square(file, rank)) {
        mask |= bit(make_square(file, rank));
        file += file_delta;
        rank += rank_delta;
    }
    return mask;
}

consteval std::array<Bitboard, BoardSize * BoardSize> make_ray_to_edge_masks() {
    std::array<Bitboard, BoardSize * BoardSize> masks{};
    for (Square from = 0; from < BoardSize; ++from) {
        for (Square through = 0; through < BoardSize; ++through) {
            masks[static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(through)] =
                make_ray_to_edge_mask(from, through);
        }
    }
    return masks;
}

constexpr auto RayToEdgeMasks = make_ray_to_edge_masks();

constexpr Bitboard make_rook_ray_to_edge_mask_or_empty(Square from, Square through) {
    if (from == through) {
        return EmptyBB;
    }
    if (file_of(from) != file_of(through) && rank_of(from) != rank_of(through)) {
        return EmptyBB;
    }
    return make_ray_to_edge_mask(from, through);
}

consteval std::array<Bitboard, BoardSize * BoardSize> make_rook_ray_to_edge_masks() {
    std::array<Bitboard, BoardSize * BoardSize> masks{};
    for (Square from = 0; from < BoardSize; ++from) {
        for (Square through = 0; through < BoardSize; ++through) {
            masks[static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(through)] =
                make_rook_ray_to_edge_mask_or_empty(from, through);
        }
    }
    return masks;
}

constexpr auto RookRayToEdgeMasks = make_rook_ray_to_edge_masks();

constexpr Bitboard make_bishop_ray_to_edge_mask_or_empty(Square from, Square through) {
    if (from == through) {
        return EmptyBB;
    }
    if (abs_int(file_of(from) - file_of(through))
        != abs_int(rank_of(from) - rank_of(through))) {
        return EmptyBB;
    }
    return make_ray_to_edge_mask(from, through);
}

consteval std::array<Bitboard, BoardSize * BoardSize> make_bishop_ray_to_edge_masks() {
    std::array<Bitboard, BoardSize * BoardSize> masks{};
    for (Square from = 0; from < BoardSize; ++from) {
        for (Square through = 0; through < BoardSize; ++through) {
            masks[static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(through)] =
                make_bishop_ray_to_edge_mask_or_empty(from, through);
        }
    }
    return masks;
}

constexpr auto BishopRayToEdgeMasks = make_bishop_ray_to_edge_masks();

Bitboard ray_to_mask(Square from, Square to) {
    assert(is_valid_square(from));
    assert(is_valid_square(to));
    return RayToMasks[static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(to)];
}

Bitboard ray_to_edge_mask(Square from, Square through) {
    assert(is_valid_square(from));
    assert(is_valid_square(through));
    return RayToEdgeMasks[
        static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(through)];
}

Bitboard rook_ray_to_edge_mask_or_empty(Square from, Square through) {
    assert(is_valid_square(from));
    assert(is_valid_square(through));
    return RookRayToEdgeMasks[
        static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(through)];
}

Bitboard bishop_ray_to_edge_mask_or_empty(Square from, Square through) {
    assert(is_valid_square(from));
    assert(is_valid_square(through));
    return BishopRayToEdgeMasks[
        static_cast<std::size_t>(from) * BoardSize + static_cast<std::size_t>(through)];
}

bool ray_increases_square_index(Square king, Bitboard ray) {
    assert(ray != EmptyBB);
    return std::countr_zero(ray) > king;
}

struct Direction {
    int file_delta = 0;
    int rank_delta = 0;
    bool rook_like = false;
};

constexpr Direction KingRayDirections[] = {
    {1, 0, true},
    {-1, 0, true},
    {0, 1, true},
    {0, -1, true},
    {1, 1, false},
    {1, -1, false},
    {-1, 1, false},
    {-1, -1, false},
};

bool is_slider_for_direction(const Position& pos, Color color, Square square, bool rook_like) {
    const Bitboard square_mask = bit(square);
    const int color_idx = static_cast<int>(color);
    if (rook_like) {
        return (pos.pieces[color_idx][static_cast<int>(PieceType::Rook)] & square_mask)
            || (pos.pieces[color_idx][static_cast<int>(PieceType::Queen)] & square_mask);
    }
    return (pos.pieces[color_idx][static_cast<int>(PieceType::Bishop)] & square_mask)
        || (pos.pieces[color_idx][static_cast<int>(PieceType::Queen)] & square_mask);
}

bool same_line(Square a, Square b, Square c) {
    const int af = file_of(a);
    const int ar = rank_of(a);
    const int bf = file_of(b);
    const int br = rank_of(b);
    const int cf = file_of(c);
    const int cr = rank_of(c);

    const int df1 = bf - af;
    const int dr1 = br - ar;
    const int df2 = cf - af;
    const int dr2 = cr - ar;
    return (df1 == 0 && df2 == 0)
        || (dr1 == 0 && dr2 == 0)
        || (std::abs(df1) == std::abs(dr1)
            && std::abs(df2) == std::abs(dr2)
            && df1 * dr2 == dr1 * df2);
}

Bitboard ray_between_exclusive(Square from, Square to) {
    const int from_file = file_of(from);
    const int from_rank = rank_of(from);
    const int to_file = file_of(to);
    const int to_rank = rank_of(to);
    const int file_delta = (to_file > from_file) - (to_file < from_file);
    const int rank_delta = (to_rank > from_rank) - (to_rank < from_rank);

    Bitboard ray = EmptyBB;
    int file = from_file + file_delta;
    int rank = from_rank + rank_delta;
    while (file != to_file || rank != to_rank) {
        ray |= bit(make_square(file, rank));
        file += file_delta;
        rank += rank_delta;
    }
    return ray;
}

Bitboard occupancy_after_move(const Position& pos, Move move) {
    const Square from = from_square(move);
    const Square to = to_square(move);
    Bitboard occupancy = pos.occupancy();
    occupancy &= ~bit(from);
    if (is_capture(move) && move_flag(move) != MoveFlag::EnPassant) {
        occupancy &= ~bit(to);
    }
    occupancy |= bit(to);
    return occupancy;
}

bool piece_attacks_square(
    PieceType piece,
    Color color,
    Square from,
    Square target,
    Bitboard occupancy
) {
    const Bitboard target_mask = bit(target);
    switch (piece) {
        case PieceType::Pawn:
            return (pawn_attacks(color, from) & target_mask) != EmptyBB;
        case PieceType::Knight:
            return (knight_attacks(from) & target_mask) != EmptyBB;
        case PieceType::Bishop:
            return (bishop_attacks(from, occupancy) & target_mask) != EmptyBB;
        case PieceType::Rook:
            return (rook_attacks(from, occupancy) & target_mask) != EmptyBB;
        case PieceType::Queen:
            return (queen_attacks(from, occupancy) & target_mask) != EmptyBB;
        case PieceType::King:
            return (king_attacks(from) & target_mask) != EmptyBB;
        case PieceType::None:
            return false;
    }
    return false;
}

} // namespace

KingSafetyContext make_king_safety_context(const Position& pos, Color color) {
    KingSafetyContext context;
    const Color us = color;
    const Color them = opposite(us);
    const int them_idx = static_cast<int>(them);
    const Bitboard king_board = pos.pieces[static_cast<int>(us)][static_cast<int>(PieceType::King)];
    assert(popcount(king_board) == 1);
    context.king_square = std::countr_zero(king_board);
    const Bitboard our_pieces = pos.occupancy(us);
    const Bitboard their_pieces = pos.occupancy(them);
    const Bitboard occupancy = our_pieces | their_pieces;
    const Bitboard enemy_rooks_or_queens =
        pos.pieces[them_idx][static_cast<int>(PieceType::Rook)]
        | pos.pieces[them_idx][static_cast<int>(PieceType::Queen)];
    const Bitboard enemy_bishops_or_queens =
        pos.pieces[them_idx][static_cast<int>(PieceType::Bishop)]
        | pos.pieces[them_idx][static_cast<int>(PieceType::Queen)];
    const Bitboard rook_attack_from_king = rook_attacks(context.king_square, occupancy);
    const Bitboard bishop_attack_from_king = bishop_attacks(context.king_square, occupancy);
    const Bitboard slider_checkers =
        (bishop_attack_from_king & enemy_bishops_or_queens)
        | (rook_attack_from_king & enemy_rooks_or_queens);

    context.checkers =
        (pawn_attacks(us, context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::Pawn)])
        | (knight_attacks(context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::Knight)])
        | slider_checkers;

    if (context.checkers == EmptyBB) {
        context.block_mask = FullBB;
    } else if (popcount(context.checkers) == 1) {
        const Square checker = std::countr_zero(context.checkers);
        const Bitboard checker_mask = bit(checker);
        if ((slider_checkers & checker_mask) != EmptyBB) {
            context.block_mask = ray_to_mask(context.king_square, checker);
        } else {
            context.block_mask = checker_mask;
        }
    } else {
        context.block_mask = EmptyBB;
    }

    const Bitboard rook_first_blockers = rook_attack_from_king & our_pieces;
    if (rook_first_blockers != EmptyBB) {
        Bitboard pinners =
            rook_attacks(context.king_square, occupancy ^ rook_first_blockers)
            & enemy_rooks_or_queens;
        while (pinners != EmptyBB) {
            const Square pinner = pop_lsb(pinners);
            context.pinned |= ray_to_mask(context.king_square, pinner) & rook_first_blockers;
        }
    }

    const Bitboard bishop_first_blockers = bishop_attack_from_king & our_pieces;
    if (bishop_first_blockers != EmptyBB) {
        Bitboard pinners =
            bishop_attacks(context.king_square, occupancy ^ bishop_first_blockers)
            & enemy_bishops_or_queens;
        while (pinners != EmptyBB) {
            const Square pinner = pop_lsb(pinners);
            context.pinned |= ray_to_mask(context.king_square, pinner) & bishop_first_blockers;
        }
    }

    return context;
}

KingSafetyContext make_king_safety_context(const Position& pos) {
    return make_king_safety_context(pos, pos.side_to_move);
}

KingSafetyContext cached_king_safety_context(const Position& pos, Color color) {
    const int color_idx = static_cast<int>(color);
    return KingSafetyContext{
        pos.king_squares[color_idx],
        pos.king_checkers[color_idx],
        pos.king_pinned[color_idx],
        pos.king_block_masks[color_idx],
    };
}

void refresh_king_safety(Position& pos) {
    for (Color color : {Color::White, Color::Black}) {
        const int color_idx = static_cast<int>(color);
        if (popcount(pos.pieces[color_idx][static_cast<int>(PieceType::King)]) != 1) {
            pos.king_squares[color_idx] = NoSquare;
            pos.king_checkers[color_idx] = EmptyBB;
            pos.king_pinned[color_idx] = EmptyBB;
            pos.king_block_masks[color_idx] = FullBB;
            continue;
        }
        const KingSafetyContext context = make_king_safety_context(pos, color);
        pos.king_squares[color_idx] = context.king_square;
        pos.king_checkers[color_idx] = context.checkers;
        pos.king_pinned[color_idx] = context.pinned;
        pos.king_block_masks[color_idx] = context.block_mask;
    }
}

namespace {

constexpr bool same_rook_line(Square a, Square b) {
    return file_of(a) == file_of(b) || rank_of(a) == rank_of(b);
}

constexpr bool same_bishop_line(Square a, Square b) {
    return abs_int(file_of(a) - file_of(b)) == abs_int(rank_of(a) - rank_of(b));
}

Square nearest_square_on_ray(Bitboard squares, bool increasing_square_index) {
    assert(squares != EmptyBB);
    return increasing_square_index
        ? std::countr_zero(squares)
        : (BoardSize - 1 - std::countl_zero(squares));
}

bool direct_piece_attacks_king(PieceType piece, Color attacker_color, Square from, Square king) {
    switch (piece) {
        case PieceType::Pawn:
            return (pawn_attacks(opposite(attacker_color), king) & bit(from)) != EmptyBB;
        case PieceType::Knight:
            return (knight_attacks(from) & bit(king)) != EmptyBB;
        case PieceType::King:
            return (king_attacks(from) & bit(king)) != EmptyBB;
        default:
            return false;
    }
}

constexpr bool is_direct_checker_piece(PieceType piece) {
    return piece == PieceType::Pawn
        || piece == PieceType::Knight
        || piece == PieceType::King;
}

Bitboard block_mask_from_checkers(const Position& pos, Color color, Square king, Bitboard checkers) {
    if (checkers == EmptyBB) {
        return FullBB;
    }
    if (popcount(checkers) != 1) {
        return EmptyBB;
    }

    const Color enemy = opposite(color);
    const int enemy_idx = static_cast<int>(enemy);
    const Square checker = std::countr_zero(checkers);
    const Bitboard checker_mask = bit(checker);
    const Bitboard enemy_rooks_or_queens =
        pos.pieces[enemy_idx][static_cast<int>(PieceType::Rook)]
        | pos.pieces[enemy_idx][static_cast<int>(PieceType::Queen)];
    const Bitboard enemy_bishops_or_queens =
        pos.pieces[enemy_idx][static_cast<int>(PieceType::Bishop)]
        | pos.pieces[enemy_idx][static_cast<int>(PieceType::Queen)];
    if ((enemy_rooks_or_queens & checker_mask) != EmptyBB
        && same_rook_line(king, checker)) {
        return ray_to_mask(king, checker);
    }
    if ((enemy_bishops_or_queens & checker_mask) != EmptyBB
        && same_bishop_line(king, checker)) {
        return ray_to_mask(king, checker);
    }
    return checker_mask;
}

void update_rook_ray(
    const Position& pos,
    Color color,
    Square king,
    Bitboard ray,
    Bitboard& checkers,
    Bitboard& pinned
) {
    assert(ray != EmptyBB);
    const Color enemy = opposite(color);
    const int enemy_idx = static_cast<int>(enemy);
    const Bitboard enemy_sliders =
        pos.pieces[enemy_idx][static_cast<int>(PieceType::Rook)]
        | pos.pieces[enemy_idx][static_cast<int>(PieceType::Queen)];
    Bitboard checker = EmptyBB;
    Bitboard pinned_piece = EmptyBB;
    const Bitboard sliders = ray & enemy_sliders;
    if (sliders != EmptyBB) {
        const Square slider = nearest_square_on_ray(sliders, ray_increases_square_index(king, ray));
        const Bitboard occupancy = pos.occupancy();
        const Bitboard our_pieces = pos.occupancy(color);
        const Bitboard blockers = ray_to_mask(king, slider) & occupancy;
        const int blocker_count = popcount(blockers);
        if (blocker_count == 0) {
            checker = bit(slider);
        } else if (blocker_count == 1 && (blockers & our_pieces) != EmptyBB) {
            pinned_piece = blockers;
        }
    }

    checkers &= ~(ray & enemy_sliders);
    checkers |= checker;
    pinned &= ~ray;
    pinned |= pinned_piece;
}

void update_bishop_ray(
    const Position& pos,
    Color color,
    Square king,
    Bitboard ray,
    Bitboard& checkers,
    Bitboard& pinned
) {
    assert(ray != EmptyBB);
    const Color enemy = opposite(color);
    const int enemy_idx = static_cast<int>(enemy);
    const Bitboard enemy_sliders =
        pos.pieces[enemy_idx][static_cast<int>(PieceType::Bishop)]
        | pos.pieces[enemy_idx][static_cast<int>(PieceType::Queen)];
    Bitboard checker = EmptyBB;
    Bitboard pinned_piece = EmptyBB;
    const Bitboard sliders = ray & enemy_sliders;
    if (sliders != EmptyBB) {
        const Square slider = nearest_square_on_ray(sliders, ray_increases_square_index(king, ray));
        const Bitboard occupancy = pos.occupancy();
        const Bitboard our_pieces = pos.occupancy(color);
        const Bitboard blockers = ray_to_mask(king, slider) & occupancy;
        const int blocker_count = popcount(blockers);
        if (blocker_count == 0) {
            checker = bit(slider);
        } else if (blocker_count == 1 && (blockers & our_pieces) != EmptyBB) {
            pinned_piece = blockers;
        }
    }

    checkers &= ~(ray & enemy_sliders);
    checkers |= checker;
    pinned &= ~ray;
    pinned |= pinned_piece;
}

void update_slider_ray(
    const Position& pos,
    Color color,
    Square king,
    Square affected_square,
    Bitboard& checkers,
    Bitboard& pinned
) {
    if (same_rook_line(king, affected_square)) {
        update_rook_ray(
            pos,
            color,
            king,
            ray_to_edge_mask(king, affected_square),
            checkers,
            pinned);
    } else if (same_bishop_line(king, affected_square)) {
        update_bishop_ray(
            pos,
            color,
            king,
            ray_to_edge_mask(king, affected_square),
            checkers,
            pinned);
    }
}

void store_king_safety(Position& pos, Color color, const KingSafetyContext& context) {
    const int color_idx = static_cast<int>(color);
    pos.king_squares[color_idx] = context.king_square;
    pos.king_checkers[color_idx] = context.checkers;
    pos.king_pinned[color_idx] = context.pinned;
    pos.king_block_masks[color_idx] = context.block_mask;
}

void recompute_cached_king_safety(Position& pos) {
    store_king_safety(pos, Color::White, make_king_safety_context(pos, Color::White));
    store_king_safety(pos, Color::Black, make_king_safety_context(pos, Color::Black));
}

} // namespace

void update_king_safety_after_move(
    Position& pos,
    Move move,
    Color moved_color,
    PieceType moved_piece,
    PieceType captured_piece,
    Square captured_square
) {
    const Square from = from_square(move);
    const Square to = to_square(move);
    const MoveFlag flag = move_flag(move);
    (void)captured_piece;

    const bool special_move =
        moved_piece == PieceType::King
        || flag == MoveFlag::KingCastle
        || flag == MoveFlag::QueenCastle
        || flag == MoveFlag::EnPassant
        || promotion_piece(move) != PieceType::None;
    if (special_move) {
        recompute_cached_king_safety(pos);
        return;
    }

    const Color enemy_color = opposite(moved_color);
    const Square enemy_king = pos.king_squares[static_cast<int>(enemy_color)];
    assert(enemy_king != NoSquare);
    const Bitboard enemy_direct_checker =
        is_direct_checker_piece(moved_piece)
            && direct_piece_attacks_king(moved_piece, moved_color, to, enemy_king)
        ? bit(to)
        : EmptyBB;

    for (Color color : {Color::White, Color::Black}) {
        const int color_idx = static_cast<int>(color);
        assert(popcount(pos.pieces[color_idx][static_cast<int>(PieceType::King)]) == 1);

        const Square king = pos.king_squares[color_idx];
        assert(king != NoSquare);
        Bitboard checkers = EmptyBB;
        Bitboard pinned = pos.king_pinned[color_idx];
        const Bitboard from_rook_ray = rook_ray_to_edge_mask_or_empty(king, from);
        const Bitboard to_rook_ray = rook_ray_to_edge_mask_or_empty(king, to);
        if (from_rook_ray != EmptyBB) {
            update_rook_ray(pos, color, king, from_rook_ray, checkers, pinned);
        }
        if (to_rook_ray != EmptyBB && to_rook_ray != from_rook_ray) {
            update_rook_ray(pos, color, king, to_rook_ray, checkers, pinned);
        }

        const Bitboard from_bishop_ray = bishop_ray_to_edge_mask_or_empty(king, from);
        const Bitboard to_bishop_ray = bishop_ray_to_edge_mask_or_empty(king, to);
        if (from_bishop_ray != EmptyBB) {
            update_bishop_ray(pos, color, king, from_bishop_ray, checkers, pinned);
        }
        if (to_bishop_ray != EmptyBB && to_bishop_ray != from_bishop_ray) {
            update_bishop_ray(pos, color, king, to_bishop_ray, checkers, pinned);
        }

        if (color == enemy_color) {
            checkers |= enemy_direct_checker;
        }

        pos.king_checkers[color_idx] = checkers;
        pos.king_pinned[color_idx] = pinned;
        pos.king_block_masks[color_idx] =
            block_mask_from_checkers(pos, color, king, checkers);
    }
}

KingSafetyContext make_old_king_safety_context(const Position& pos) {
    KingSafetyContext context;
    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const int them_idx = static_cast<int>(them);
    const Bitboard king_board = pos.pieces[static_cast<int>(us)][static_cast<int>(PieceType::King)];
    assert(popcount(king_board) == 1);
    context.king_square = std::countr_zero(king_board);
    const Bitboard our_pieces = pos.occupancy(us);
    const Bitboard their_pieces = pos.occupancy(them);
    const Bitboard occupancy = our_pieces | their_pieces;
    const Bitboard enemy_rooks_or_queens =
        pos.pieces[them_idx][static_cast<int>(PieceType::Rook)]
        | pos.pieces[them_idx][static_cast<int>(PieceType::Queen)];
    const Bitboard enemy_bishops_or_queens =
        pos.pieces[them_idx][static_cast<int>(PieceType::Bishop)]
        | pos.pieces[them_idx][static_cast<int>(PieceType::Queen)];
    const Bitboard rook_attack_from_king = rook_attacks(context.king_square, occupancy);
    const Bitboard bishop_attack_from_king = bishop_attacks(context.king_square, occupancy);

    context.checkers =
        (pawn_attacks(us, context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::Pawn)])
        | (knight_attacks(context.king_square)
            & pos.pieces[them_idx][static_cast<int>(PieceType::Knight)])
        | (bishop_attack_from_king & enemy_bishops_or_queens)
        | (rook_attack_from_king & enemy_rooks_or_queens);

    if (context.checkers == EmptyBB) {
        context.block_mask = FullBB;
    } else if (popcount(context.checkers) == 1) {
        const Square checker = std::countr_zero(context.checkers);
        if (same_line(context.king_square, checker, checker)
            && is_slider_for_direction(
                pos,
                them,
                checker,
                file_of(context.king_square) == file_of(checker)
                    || rank_of(context.king_square) == rank_of(checker))) {
            context.block_mask = ray_between_exclusive(context.king_square, checker) | bit(checker);
        } else {
            context.block_mask = bit(checker);
        }
    } else {
        context.block_mask = EmptyBB;
    }

    for (const Direction direction : KingRayDirections) {
        Square blocker = NoSquare;
        int file = file_of(context.king_square) + direction.file_delta;
        int rank = rank_of(context.king_square) + direction.rank_delta;
        while (is_valid_square(file, rank)) {
            const Square square = make_square(file, rank);
            const Bitboard square_mask = bit(square);
            if (our_pieces & square_mask) {
                if (blocker != NoSquare) {
                    break;
                }
                blocker = square;
            } else if (their_pieces & square_mask) {
                if (blocker != NoSquare
                    && is_slider_for_direction(pos, them, square, direction.rook_like)) {
                    context.pinned |= bit(blocker);
                }
                break;
            }
            file += direction.file_delta;
            rank += direction.rank_delta;
        }
    }

    return context;
}

bool is_pseudo_move_legal(
    const Position& pos,
    const KingSafetyContext& king_safety,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece
) {
    if (move_flag(move) == MoveFlag::EnPassant) {
        Position next = pos;
        next.make_move(move, moved_piece, captured_piece);
        return !in_check(next, pos.side_to_move);
    }

    const Square from = from_square(move);
    const Square to = to_square(move);
    if (moved_piece == PieceType::King) {
        if (move_flag(move) == MoveFlag::KingCastle || move_flag(move) == MoveFlag::QueenCastle) {
            if (king_safety.checkers != EmptyBB) {
                return false;
            }
            const int direction = move_flag(move) == MoveFlag::KingCastle ? 1 : -1;
            const Square transit = make_square(file_of(from) + direction, rank_of(from));
            if (is_square_attacked(pos, transit, opposite(pos.side_to_move))) {
                return false;
            }
        }

        Bitboard occupancy = pos.occupancy();
        occupancy &= ~bit(from);
        Square excluded_attacker_square = NoSquare;
        if (is_capture(move)) {
            occupancy &= ~bit(to);
            excluded_attacker_square = to;
        }
        occupancy |= bit(to);
        return !is_square_attacked(
            pos,
            to,
            opposite(pos.side_to_move),
            occupancy,
            excluded_attacker_square);
    }

    if (popcount(king_safety.checkers) >= 2) {
        return false;
    }
    if (king_safety.checkers != EmptyBB
        && (king_safety.block_mask & bit(to)) == EmptyBB) {
        return false;
    }
    if ((king_safety.pinned & bit(from)) != EmptyBB
        && !same_line(king_safety.king_square, from, to)) {
        return false;
    }

    return true;
}

bool gives_check_fast(
    const Position& pos,
    Move move,
    PieceType moved_piece,
    PieceType captured_piece
) {
    const MoveFlag flag = move_flag(move);
    if (flag == MoveFlag::EnPassant
        || flag == MoveFlag::KingCastle
        || flag == MoveFlag::QueenCastle) {
        Position next = pos;
        next.make_move(move, moved_piece, captured_piece);
        return in_check(next, next.side_to_move);
    }

    const Color us = pos.side_to_move;
    const Color them = opposite(us);
    const Square from = from_square(move);
    const Square to = to_square(move);
    const Square enemy_king = king_square(pos, them);
    const Bitboard occupancy = occupancy_after_move(pos, move);

    PieceType checking_piece = promotion_piece(move);
    if (checking_piece == PieceType::None) {
        checking_piece = moved_piece;
    }
    if (piece_attacks_square(checking_piece, us, to, enemy_king, occupancy)) {
        return true;
    }

    const int us_idx = static_cast<int>(us);
    const Bitboard from_mask = bit(from);
    const Bitboard bishops_or_queens =
        (pos.pieces[us_idx][static_cast<int>(PieceType::Bishop)]
            | pos.pieces[us_idx][static_cast<int>(PieceType::Queen)])
        & ~from_mask;
    const Bitboard rooks_or_queens =
        (pos.pieces[us_idx][static_cast<int>(PieceType::Rook)]
            | pos.pieces[us_idx][static_cast<int>(PieceType::Queen)])
        & ~from_mask;

    return (bishop_attacks(enemy_king, occupancy) & bishops_or_queens) != EmptyBB
        || (rook_attacks(enemy_king, occupancy) & rooks_or_queens) != EmptyBB;
}

} // namespace chess::experimental
