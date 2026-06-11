#include "board_encoder.hpp"

#include "move.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

using namespace chess;

namespace {

bool contains_feature(const EncodedPosition& encoded, FeatureIndex feature) {
    return std::binary_search(encoded.features.begin(), encoded.features.end(), feature);
}

int aux_sum(const EncodedPosition& encoded) {
    return std::accumulate(encoded.aux.begin(), encoded.aux.end(), 0);
}

int piece_count(const Position& pos) {
    return popcount(pos.occupancy());
}

void assert_encoder_invariants(const Position& pos) {
    const EncodedPosition encoded = encode_position(pos);

    assert(encoded.features.size() == static_cast<std::size_t>(piece_count(pos) * 2));
    assert(std::is_sorted(encoded.features.begin(), encoded.features.end()));
    assert(std::adjacent_find(encoded.features.begin(), encoded.features.end()) == encoded.features.end());

    for (FeatureIndex feature : encoded.features) {
        assert(feature < EncodedFeatureCount);
    }

    for (std::uint8_t value : encoded.aux) {
        assert(value == 0 || value == 1);
    }

    int ep_file_count = 0;
    for (int file = 0; file < 8; ++file) {
        ep_file_count += encoded.aux[EnPassantFileA + file];
    }
    assert(ep_file_count == (encoded.aux[HasEnPassant] ? 1 : 0));
}

} // namespace

int main() {
    {
        static_assert(EncodedFeatureCount == 6 * 2 * 2 * 64 * 64);
        static_assert(feature_index(PieceType::Pawn,
                                    EncodedPieceSide::Friendly,
                                    EncodedKingContext::FriendlyKing,
                                    0,
                                    0) == 0);
        static_assert(feature_index(PieceType::King,
                                    EncodedPieceSide::Enemy,
                                    EncodedKingContext::EnemyKing,
                                    63,
                                    63) == EncodedFeatureCount - 1);
    }

    {
        Position pos;
        pos.set_startpos();

        const EncodedPosition encoded = encode_position(pos);

        assert_encoder_invariants(pos);
        assert(encoded.features.size() == 64);
        assert(std::adjacent_find(encoded.features.begin(), encoded.features.end()) == encoded.features.end());
        assert(encoded.features.front() < EncodedFeatureCount);
        assert(encoded.features.back() < EncodedFeatureCount);
        assert(encoded.aux[FriendlyCanCastleKingside] == 1);
        assert(encoded.aux[FriendlyCanCastleQueenside] == 1);
        assert(encoded.aux[EnemyCanCastleKingside] == 1);
        assert(encoded.aux[EnemyCanCastleQueenside] == 1);
        assert(encoded.aux[HasEnPassant] == 0);
        assert(aux_sum(encoded) == 4);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(6, 7));
        pos.side_to_move = Color::White;

        const EncodedPosition encoded = encode_position(pos);
        assert_encoder_invariants(pos);
        const Square friendly_king = make_square(4, 0);
        const Square enemy_king = make_square(4, 7);

        assert(contains_feature(encoded,
            feature_index(PieceType::Pawn,
                          EncodedPieceSide::Friendly,
                          EncodedKingContext::FriendlyKing,
                          friendly_king,
                          make_square(4, 1))));
        assert(contains_feature(encoded,
            feature_index(PieceType::Pawn,
                          EncodedPieceSide::Friendly,
                          EncodedKingContext::EnemyKing,
                          enemy_king,
                          make_square(4, 1))));
        assert(contains_feature(encoded,
            feature_index(PieceType::Knight,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::FriendlyKing,
                          friendly_king,
                          make_square(6, 7))));
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 1));
        pos.set_piece(Color::Black, PieceType::Knight, make_square(6, 7));
        pos.side_to_move = Color::Black;
        pos.black_can_castle_kingside = true;
        pos.white_can_castle_queenside = true;
        pos.en_passant_square = make_square(3, 2);

        const EncodedPosition encoded = encode_position(pos);
        assert_encoder_invariants(pos);
        const Square friendly_king = make_square(4, 0);
        const Square enemy_king = make_square(4, 7);

        assert(contains_feature(encoded,
            feature_index(PieceType::Knight,
                          EncodedPieceSide::Friendly,
                          EncodedKingContext::FriendlyKing,
                          friendly_king,
                          make_square(6, 0))));
        assert(contains_feature(encoded,
            feature_index(PieceType::Pawn,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::EnemyKing,
                          enemy_king,
                          make_square(4, 6))));

        assert(encoded.aux[FriendlyCanCastleKingside] == 1);
        assert(encoded.aux[FriendlyCanCastleQueenside] == 0);
        assert(encoded.aux[EnemyCanCastleKingside] == 0);
        assert(encoded.aux[EnemyCanCastleQueenside] == 1);
        assert(encoded.aux[HasEnPassant] == 1);
        assert(encoded.aux[EnPassantFileA + 3] == 1);
        assert(aux_sum(encoded) == 4);
    }

    {
        Position pos;
        pos.set_startpos();
        pos.make_move(make_move(make_square(4, 1), make_square(4, 3), MoveFlag::DoublePawnPush));

        const EncodedPosition encoded = encode_position(pos);
        assert_encoder_invariants(pos);

        const Square friendly_king = make_square(4, 0);
        const Square enemy_king = make_square(4, 7);

        assert(pos.side_to_move == Color::Black);
        assert(contains_feature(encoded,
            feature_index(PieceType::Pawn,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::FriendlyKing,
                          friendly_king,
                          make_square(4, 4))));
        assert(contains_feature(encoded,
            feature_index(PieceType::Pawn,
                          EncodedPieceSide::Friendly,
                          EncodedKingContext::EnemyKing,
                          enemy_king,
                          make_square(4, 1))));
        assert(encoded.aux[HasEnPassant] == 1);
        assert(encoded.aux[EnPassantFileA + 4] == 1);
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));
        pos.make_move(make_move(make_square(4, 0), make_square(6, 0), MoveFlag::KingCastle));

        const EncodedPosition encoded = encode_position(pos);
        assert_encoder_invariants(pos);

        assert(pos.side_to_move == Color::Black);
        assert(contains_feature(encoded,
            feature_index(PieceType::King,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::EnemyKing,
                          make_square(6, 7),
                          make_square(6, 7))));
        assert(contains_feature(encoded,
            feature_index(PieceType::Rook,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::FriendlyKing,
                          make_square(4, 0),
                          make_square(5, 7))));
        assert(encoded.aux[FriendlyCanCastleKingside] == 1);
        assert(encoded.aux[FriendlyCanCastleQueenside] == 1);
        assert(encoded.aux[EnemyCanCastleKingside] == 0);
        assert(encoded.aux[EnemyCanCastleQueenside] == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(4, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 4));
        pos.set_piece(Color::Black, PieceType::Pawn, make_square(3, 4));
        pos.side_to_move = Color::White;
        pos.en_passant_square = make_square(3, 5);

        pos.make_move(make_move(make_square(4, 4), make_square(3, 5), MoveFlag::EnPassant));

        const EncodedPosition encoded = encode_position(pos);
        assert_encoder_invariants(pos);

        assert(pos.is_empty(make_square(3, 4)));
        assert(piece_count(pos) == 3);
        assert(encoded.features.size() == 6);
        assert(encoded.aux[HasEnPassant] == 0);
    }

    {
        Position pos;
        pos.set_piece(Color::White, PieceType::King, make_square(4, 0));
        pos.set_piece(Color::Black, PieceType::King, make_square(0, 7));
        pos.set_piece(Color::White, PieceType::Pawn, make_square(4, 6));
        pos.side_to_move = Color::White;

        pos.make_move(make_move(make_square(4, 6), make_square(4, 7), MoveFlag::QueenPromotion));

        const EncodedPosition encoded = encode_position(pos);
        assert_encoder_invariants(pos);

        assert(contains_feature(encoded,
            feature_index(PieceType::Queen,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::FriendlyKing,
                          make_square(0, 0),
                          make_square(4, 0))));
        assert(!contains_feature(encoded,
            feature_index(PieceType::Pawn,
                          EncodedPieceSide::Enemy,
                          EncodedKingContext::FriendlyKing,
                          make_square(0, 0),
                          make_square(4, 0))));
    }

    {
        Position pos;
        pos.set_startpos();
        assert_encoder_invariants(pos);

        const std::vector<Move> first_moves = generate_legal_moves(pos);
        for (Move first : first_moves) {
            Position after_first = pos;
            after_first.make_move(first);
            assert_encoder_invariants(after_first);

            const std::vector<Move> second_moves = generate_legal_moves(after_first);
            for (Move second : second_moves) {
                Position after_second = after_first;
                after_second.make_move(second);
                assert_encoder_invariants(after_second);
            }
        }
    }

    {
        Position pos;
        assert(pos.set_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));
        assert_encoder_invariants(pos);

        const std::vector<Move> moves = generate_legal_moves(pos);
        for (Move move : moves) {
            Position next = pos;
            next.make_move(move);
            assert_encoder_invariants(next);
        }
    }

    {
        Position pos;
        pos.set_startpos();

        for (int ply = 0; ply < 80; ++ply) {
            assert_encoder_invariants(pos);
            const std::vector<Move> moves = generate_legal_moves(pos);
            if (moves.empty()) {
                break;
            }

            const std::size_t index = static_cast<std::size_t>((ply * 17 + moves.size() * 5) % moves.size());
            pos.make_move(moves[index]);
        }
        assert_encoder_invariants(pos);
    }
}
