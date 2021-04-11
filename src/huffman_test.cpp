#include "huffman_test.hpp"

int main(int argc, char* argv[])
{
	huffman_args_info info = ParseArgs(static_cast<size_t>(argc) - 1, argv + 1);
	std::vector<CharT> input = ReadEntireFile<CharT>(info.input);
	std::vector<CharT> output;
	std::string output_path;

	if (info.output.empty())
	{
		constexpr std::string_view extension{ ".huffPP" };
		output_path = info.input.substr(0, info.input.find_last_of('.'));
		output_path.insert(output_path.end(), extension.begin(), extension.end());
		info.output = output_path;
	}

	std::string_view input_filename = GetPathFilename(info.input);
	std::string_view output_filename = GetPathFilename(info.output);
	const size_t input_len = input.size();

	printf("\n[+] Input Filename: %s\n", input_filename.data());
	printf("[+] Input Size: %zu bytes | %.2f kilobytes | %.2f megabytes\n",
			input_len, static_cast<double>(input_len) / 1024.0,
			static_cast<double>(input_len) / (1024.0 * 1024.0));

	auto timelapse = Bench([&info, &input, &output]()
	{
		if (info.method == huffman_method::encode)
		{
			huffmanPP::HuffmanCompress<CharT>(std::make_move_iterator(input.begin()),
											  std::make_move_iterator(input.end()),
											  std::back_inserter(output));
		}
		else
		{
			huffmanPP::HuffmanDecompress<CharT>(std::make_move_iterator(input.begin()),
												std::make_move_iterator(input.end()),
												std::back_inserter(output));
		}
		
		WriteBufferToFile(output, info.output);
	});

	const huffPP::details::ssize_t r1 = input.size() - output.size();
	double rest_size = (r1 < 0) ? 0.0 : static_cast<double>(r1);
	size_t output_len = output.size();

	if (info.method == huffman_method::encode)
	{
		double compression_ratio = (rest_size * 100.0) / static_cast<double>(input.size());
		printf("[+] Compression Ratio: %.2f%%\n", compression_ratio);
	}

	printf("[+] Output Filename: %s\n", output_filename.data());
	
	printf("[+] Output Size: %zu bytes | %.2f kilobytes | %.2f megabytes\n",
		   output_len, static_cast<double>(output_len) / 1024,
		   static_cast<double>(output_len) / (1024 * 1024));

	printf("[+] Succesfully\n");
	printf("[+] Times Taken: %.4f seconds\n", timelapse);

	putchar('\n');
}

