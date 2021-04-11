#ifndef HUFFMAN_TEST_HPP
#define HUFFMAN_TEST_HPP

#include <cstdio>
#include <cassert>
#include <cstring>
#include <chrono>
#include <vector>
#include <array>
#include <string_view>
#include "huffman_coding.hpp"

#define TimeNow() std::chrono::steady_clock::now()

using huffmanPP::detail::Ensure;
using CharT = char8_t;

namespace huffPP::details
{
	using ssize_t = std::make_signed_t<size_t>;
}

enum class huffman_method
{
	encode,
	decode,
	unknown
};

struct huffman_args_info
{
	std::string_view input;
	std::string_view output;
	huffman_method method{ huffman_method::unknown };
};

void PrintMenu() noexcept
{
	putchar('\n');
	printf(
		"[Description]\n"
		"The huffPP (Huffman Coding Plus Plus) is a utility tool written in C++ 20 \n"
		"to compact and unzip files using the huffman coding algorithm.\n\n"

		"[Author]\n"
		"- Felipe Garcia (https://github.com/fgarcia0x0/)\n\n"

		"[Usage]:\n"
		"huffPP.exe [options]\n\n"

		"[options]:\n"
		"  -i, --input   : The input file\n"
		"  -o, --output  : The output file\n"
		"  -e, --encode  : Compressed the input file and writes the result in the output file\n"
		"  -d, --decode  : Uncompress the input file and writes the result in the output file\n"
		"  -v, --version : The program version\n\n"

		"[Examples]:\n\n"

		"[Windows]\n"
		"[Compress File]\n"
		"huffPP.exe --input input_test.txt --encode --output input_test.huffPP\n\n"

		"[Decompress File]\n"
		"huffPP.exe --input input_test.huffPP --decode --output input_test_decompressed.txt\n\n"

		"[Mac && Linux]\n"
		"[Compress File]\n"
		"./huffPP --input input_test.txt --encode --output input_test.huffPP\n\n"

		"[Decompress File]\n"
		"./huffPP --input input_test.huffPP --decode --output input_test_decompressed.txt\n"
	);

	putchar('\n');
}

using namespace std::string_view_literals;

static constexpr std::array<std::array<std::string_view, 2U>, 5U> option_table = 
{{
	{"-i"sv, "--input"sv},
	{"-o"sv, "--output"sv},
	{"-e"sv, "--encode"sv},
	{"-d"sv, "--decode"sv},
	{"-v"sv, "--version"sv}
}};

constexpr inline bool Contains(const auto& conteiner, const auto& element)
{
	for (const auto& curr_elem : conteiner)
	{
		if (curr_elem == element)
		{
			return true;
		}
	}
	return false;
}

constexpr auto ParseArgs(size_t argc, char* argv[])
{
	if (!argc || !argv)
	{
		PrintMenu();
		exit(EXIT_SUCCESS);
	}

	huffman_args_info args_info{  };
	std::string_view arg{ };
	const char* next_arg{ nullptr };

	if (argc == 1U && Contains(option_table[4], argv[0]))
	{
		printf("[+] huffPP version 0.0.1\n\n");
		exit(EXIT_SUCCESS);
	}

	for (size_t i = 0; i < argc; ++i)
	{
		arg = argv[i];

		if (Contains(option_table[0], arg))
		{
			next_arg = argv[++i];
			Ensure(next_arg && *next_arg, "[-] The option requires at least one argument");
			args_info.input = next_arg;
		}
		else if (Contains(option_table[1], arg))
		{
			next_arg = argv[++i];
			Ensure(next_arg && *next_arg, "[-] The option requires at least one argument");
			args_info.output = next_arg;
		}
		else if (Contains(option_table[2], arg))
			args_info.method = huffman_method::encode;
		else if (Contains(option_table[3], arg))
			args_info.method = huffman_method::decode;
		else
		{
			fprintf(stderr, "[-] The program does not recognize the command \"%.32s\" ", arg.data());
			exit(EXIT_FAILURE);
		}
	}

	Ensure(args_info.method != huffman_method::unknown,
		   "[-] You need to specify the method (encode/decode)");

	if (args_info.method == huffman_method::decode && args_info.output.empty())
		Ensure(false, "[-] You need to specify the output path");

	return args_info;
}

template<typename F, typename... Args>
inline double Bench(F func, Args&&... args)
{
	auto t1 = TimeNow();
	func(std::forward<Args>(args)...);
	return std::chrono::duration<double>(TimeNow() - t1).count();
}

template <typename T>
static std::vector<T> ReadEntireFile(std::string_view filepath)
{
	FILE* fileptr = fopen(filepath.data(), "rb");
	Ensure(fileptr, "[-] Could not open the file (The most common reason: invalid file path)");

	fseek(fileptr, 0, SEEK_END);
#ifdef _WIN32
	size_t len = static_cast<size_t>(_ftelli64(fileptr));
#else
	size_t len = static_cast<size_t>(ftell(fileptr));
#endif
	fseek(fileptr, 0, SEEK_SET);

	Ensure(len > 0, "[-] Could not open the file (The most common reason: invalid file path)");

	std::vector<T> buffer(len);
	(void) fread(buffer.data(), len, sizeof(T), fileptr);

	fclose(fileptr);
	return buffer;
}

template <typename T>
static void WriteBufferToFile(const std::vector<T>& vec, std::string_view filename)
{
	FILE* fileptr = fopen(filename.data(), "wb");
	Ensure(fileptr, "[-] Could not open the file for write");

	size_t nread = fwrite(vec.data(), vec.size(), sizeof(T), fileptr);
	Ensure(nread == sizeof(T), "[-] Error to write file");

	fclose(fileptr);
}

constexpr inline auto GetPathFilename(const std::string_view path)
{
	auto pos = path.find_last_of('\\');
	if (pos == std::string_view::npos)
		pos = path.find_last_of('/');

	if (pos == std::string_view::npos)
		return path;

	return path.substr(pos + 1);
}

#endif
