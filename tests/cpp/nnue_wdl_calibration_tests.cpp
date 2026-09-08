#include "nnue_wdl_calibration.hpp"

#include <cassert>
#include <cmath>

int main() {
    using chess::wdl_calibration::expected_score;
    using chess::wdl_calibration::expected_score_loss;
    using chess::wdl_calibration::probabilities;

    for (int phase = 0; phase < 8; ++phase) {
        for (int ply = 0; ply <= 240; ply += 12) {
            double previous = -1.0;
            for (int cp = -2000; cp <= 2000; cp += 25) {
                const auto wdl = probabilities(cp, phase, ply);
                const auto mirrored = probabilities(-cp, phase, ply);
                assert(std::abs(wdl.loss - mirrored.win) < 1e-12);
                assert(std::abs(wdl.draw - mirrored.draw) < 1e-12);
                assert(std::abs(wdl.win - mirrored.loss) < 1e-12);
                assert(std::abs(wdl.loss + wdl.draw + wdl.win - 1.0) < 1e-12);
                const double score = expected_score(cp, phase, ply);
                assert(score + 1e-12 >= previous);
                previous = score;
            }
        }
    }

    assert(expected_score_loss(300, 300, 4, 80) == 0.0);
    assert(expected_score_loss(300, -300, 4, 80) > 0.25);
    assert(std::abs(
        expected_score_loss(300, -100, 4, 80)
        - expected_score_loss(-300, 100, 4, 80)) < 1e-12);
}
