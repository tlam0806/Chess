#define CHESS_UCI_NNUE_V44
#define main uci_nnue_v44_cli_main_for_test
#include "../../tools/ops/uci_nnue_v38.cpp"
#undef main

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: uci_nnue_v44_cli_tests <model>\n";
        return 2;
    }

    std::istringstream commands{"uci\nquit\n"};
    std::ostringstream output;
    std::ostringstream errors;
    std::streambuf* const old_input = std::cin.rdbuf(commands.rdbuf());
    std::streambuf* const old_output = std::cout.rdbuf(output.rdbuf());
    std::streambuf* const old_errors = std::cerr.rdbuf(errors.rdbuf());

    std::string executable = "uci_nnue_v44";
    std::string model = argv[1];
    char* cli_argv[]{executable.data(), model.data()};
    const int exit_code = uci_nnue_v44_cli_main_for_test(2, cli_argv);

    std::cin.rdbuf(old_input);
    std::cout.rdbuf(old_output);
    std::cerr.rdbuf(old_errors);

    assert(exit_code == 0);
    assert(errors.str().empty());
    const std::string text = output.str();
    assert(text.find("id name ChessNNUEV44\n") != std::string::npos);
    assert(
        text.find("info string tt_lifecycle=clear_each_search\n")
        != std::string::npos);
    assert(
        text.find("info string tt_generation=none\n")
        != std::string::npos);
    assert(text.find("option name ReuseStaleTtScores") == std::string::npos);
    assert(
        text.find(
            "option name ReuseDeeperTtScores type check default false\n")
        != std::string::npos);
    assert(
        text.find("option name LmrDivisor type string default 2.9\n")
        != std::string::npos);
    assert(
        text.find(
            "option name AspirationDeltaDivisor type spin default 33700 ")
        != std::string::npos);
    assert(text.ends_with("uciok\n"));

    std::cout << "uci nnue v44 CLI tests passed\n";
    return 0;
}
