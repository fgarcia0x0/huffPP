#ifndef HUFFMAN_CODING_HPP
#define HUFFMAN_CODING_HPP

#include <iostream>
#include <cstdio>
#include <iterator>
#include <string_view>
#include <cstdint>
#include <vector>
#include <map>
#include <unordered_map>
#include <queue>
#include <type_traits>
#include <cstring>

/**
 * @brief This namespace is intended only for internal types and settings
 * to huffmanPP
 */
namespace huffmanPP::detail
{
	template <typename Iter>
	struct iter_value_type
	{
		using type = std::iter_value_t<Iter>;
	};

	template <typename Container>
	struct iter_value_type<std::back_insert_iterator<Container>>
	{
		using type = typename Container::value_type;
	};

	template <typename Container>
	struct iter_value_type<std::front_insert_iterator<Container>>
	{
		using type = typename Container::value_type;
	};

	template <typename CharType>
	using observer = std::basic_string_view<CharType>;

	template <typename CharType>
	using freq_table_type = std::vector<std::pair<CharType, size_t>>;

	using code_word_type = std::vector<bool>;

	template <typename CharType>
	using code_table_type = std::unordered_map<CharType, code_word_type>;

	constexpr auto GetNumberOfNodes = [](auto* root_node)
	{
		using node_type = std::remove_pointer_t<decltype(root_node)>;
		std::queue<node_type *> queue_node;

		size_t node_cnt{ 0 };
		size_t leaf_cnt{ 0 };

		queue_node.push(root_node);

		while (!queue_node.empty())
		{
			node_type* temp = queue_node.front();
			queue_node.pop();

			if (temp->IsLeafNode())
				++leaf_cnt;

			++node_cnt;

			if (temp->left)
				queue_node.push(temp->left);

			if (temp->right)
				queue_node.push(temp->right);
		}

		return std::make_pair(node_cnt, leaf_cnt);
	};

	template <typename T, typename OutIter>
	constexpr auto inline WriteObjInMem(const T& obj, OutIter out, 
										const size_t count = 1U)
	{
		const uint8_t* ptr_begin = nullptr;

		if constexpr (std::is_pointer_v<T>)
			ptr_begin = reinterpret_cast<const uint8_t *>(obj);
		else
			ptr_begin = reinterpret_cast<const uint8_t *>(&obj);
			
		const auto* ptr_end = ptr_begin + sizeof(T) * count;
		std::copy(ptr_begin, ptr_end, out);
	}

	template <typename Iter, typename ObjType>
	constexpr void inline ReadObjInMem(Iter mem, ObjType& obj,
									   const size_t count = 1U)
	{
		using T = std::remove_cvref_t<ObjType>;

		std::copy(mem, mem + (sizeof(T) * count),
				  reinterpret_cast<uint8_t *>(&obj));
	}

	template<typename T, typename... U>
	concept IsAnyOf = (std::same_as<T, U> || ...);

	template <std::integral I>
	constexpr inline bool IsPowerOfTwo(I x)
	{
		return !!x && !(x & (x - 1));
	}

	/**
	 * @brief Write all bit vector bits in out (Output Iterator)
	 * @tparam CharType The "character" type to be encoded/decoded
	 * @tparam OutIter  The type of output iterator
	 * @param vec_bits  A bit vector (code_word_type)(e.g std::vector <bool>)
	 * @param out An output iterator
	 */
	template <typename CharType, typename OutIter>
	auto WriteCodewordTo(code_word_type vec_bits, OutIter out)
	{
		constexpr auto MAX_BITS_OF_TYPE{ sizeof(CharType) << 3U };

		static_assert(IsPowerOfTwo(MAX_BITS_OF_TYPE), 
					  "max bits of type \"CharType\" not is power of two");

		const uint8_t nbits = vec_bits.size() & (MAX_BITS_OF_TYPE - 1);
		*out++ = nbits;

		uint8_t byte = 0;
		const size_t vec_size = vec_bits.size();

		for (size_t i = 0; i < vec_size; i++)
		{
			unsigned offset = i & (MAX_BITS_OF_TYPE - 1);
			byte |= static_cast<uint8_t>(static_cast<uint8_t>(vec_bits[i]) << offset);

			if (offset == (MAX_BITS_OF_TYPE - 1))
			{
				*out++ = byte;
				byte = 0;
			}
		}

		if (nbits)
			*out++ = byte;
	}

	/**
	 * @brief Extract the bits from each entry byte in the range [beg, end)
	 * 
	 * @tparam CharType The "character" type to be encoded/decoded
	 * @tparam InputIter The type of "iterator" of the range
	 * @param beg An "iterator" for the beginning of container/memory.
	 * @param end An "iterator" for the end of container/memory.
	 * @return A bit vector (i.e code_word_type) extracted from each byte in the range.
	 */
	template <typename CharType, typename InputIter>
	code_word_type ToCodeword(InputIter beg, InputIter end)
	{
		constexpr unsigned MAX_BITS_OF_TYPE{ sizeof(CharType) << 3U };

		static_assert(IsPowerOfTwo(MAX_BITS_OF_TYPE), 
					  "max bits of type \"CharType\" not is power of two");

		code_word_type result;
		const size_t max_len = std::distance(beg, end);

		if (max_len == 1)
			return result;

		/**
		 * @brief We reserve memory for the maximum number of code words
		 * 		  that the result will contain
		 * 
		 */
		result.reserve(max_len * MAX_BITS_OF_TYPE);
		
		const auto nbits = *beg;
		for (auto it = beg + 1; it != end; ++it)
		{
			for (unsigned i = 0; i < MAX_BITS_OF_TYPE; ++i)
			{
				result.push_back((*it >> i) & 1U);
			}
		}

		if (nbits)
		{
			for (unsigned i = 0; i < (MAX_BITS_OF_TYPE - nbits); ++i)
			{
				result.pop_back();
			}
		}

		return result;
	}

	template <typename T, size_t N> requires (N > 3U)
	consteval uint32_t FourCC(const T(&str)[N])
	{
		return static_cast<uint32_t>(
			(str[3] << 24U) | 
			(str[2] << 16U) | 
			(str[1] << 8U)  |
			(str[0]		 )
		);
	}

	constexpr inline auto Ensure(bool condition, const char* msg)
	{
		if (!condition)
		{
			fprintf(stderr, "%s\n", msg);
			exit(EXIT_FAILURE);
		}
	}

}

namespace huffmanPP
{
	/**
	 * @brief signature of our compression method, every file compressed will contain
	 * in their early bytes the signature "Huff" so that we can only evaluate
	 * The compressed files with our compression algorithm.
	 */
	constinit auto HuffmanSignature{ detail::FourCC("HUFF") };

	template <typename CharType = std::uint8_t>
	struct HuffmanTreeNode
	{
		CharType ch;
		size_t frequency;
		HuffmanTreeNode* left;
		HuffmanTreeNode* right;

		HuffmanTreeNode(CharType ch_, size_t freq_,
						HuffmanTreeNode* lptr = nullptr,
						HuffmanTreeNode* rptr = nullptr)
			: ch{ ch_ }
			, frequency{ freq_ }
			, left{ lptr }
			, right{ rptr }
		{
		}
		
		/**
		 * @brief Checks if this node is a leaf node (which is a node that has no "children")
		 * 
		 * @return true if it is a leaf node, otherwise returns false
		 */
		constexpr inline bool IsLeafNode() const noexcept
		{
			return !left && !right;
		}

		/**
		 * @brief A comparison functor, this will be used in the priority queue
		 * 		  so that we can order the nodes through frequency
		 */
		struct Comparator
		{
			constexpr inline bool operator()(HuffmanTreeNode* l, HuffmanTreeNode* r)
			{
				return l->frequency > r->frequency;
			}
		};
	};

	template <typename CharType = std::uint8_t>
	class HuffmanTree
	{
	public:
		using value_type = CharType;
		using node_type  = HuffmanTreeNode<value_type>;
	public:
		HuffmanTree() : m_root{}, m_length{}, m_leaf_count{}
		{
		}

		HuffmanTree(node_type* root_, size_t length_ = 0, size_t leaf_count = 0)
			: m_root{ root_ }, m_length{ length_ }, m_leaf_count{ leaf_count }
		{
		}

		/**
		 * @brief Get the root node of a huffman tree
		 * 
		 * @return A constant reference of root node pointer
		 */
		constexpr inline auto& GetRootNode() const noexcept
		{
			return m_root;
		}

		/**
		 * @brief Get the root node of the huffman tree
		 * 
		 * @return A root node pointer reference
		 */
		constexpr inline auto& GetRootNode() noexcept
		{
			return m_root;
		}

		/**
		 * @brief Get the amount of nodes in this tree
		 * 
		 * @return The amount of nodes of this tree
		 */
		constexpr inline size_t GetLength() const noexcept
		{
			return m_length;
		}

		/**
		 * @brief Get the amount of leaf nodes of this tree
		 * 
		 * @return The number of leaf nodes of this tree
		 */
		constexpr inline size_t GetLeafCount() const noexcept
		{
			return m_leaf_count;
		}

		/**
		 * @brief Gets the size in bytes of each present node in this tree
		 * 
		 * @return Sum size in bytes of each present node in this tree
		 */
		constexpr inline size_t GetSizeInBytes() const noexcept
		{
			return m_length * sizeof(node_type);
		}

		/**
		 * @brief Release from memory (deallocate) all nodes present in the tree
		 * 
		 */
		constexpr inline auto Release()
		{
			using namespace detail;

			std::queue<node_type *> qnodes;
			qnodes.push(m_root);

			while (!qnodes.empty())
			{
				auto* temp = qnodes.front();
				qnodes.pop();
			
				if (temp->left)
					qnodes.push(temp->left);
				
				if (temp->right)
					qnodes.push(temp->right);

				delete temp;
			}

			m_root = nullptr;
		}

		inline ~HuffmanTree()
		{
			Release();
		}

	private:
		node_type* m_root;
		size_t m_length;
		size_t m_leaf_count;
	};

	/**
	 * @brief Builds the frequency table from the given entrance
	 * 
	 * @tparam CharType The type of "character" that will be encoded/decoded
	 * @param begin An "iterator" for the beginning of the container/memory
	 * @param end   An "iterator" for the end of the container/memory
	 * @return a "pairs" vector of type (character, frequency)
	 */
	template <typename CharType, std::input_iterator Iter>
	constexpr inline auto BuildFrequencyTable(Iter begin, Iter end)
	{
		/**
		 * 
		 * @brief Here we use a hashtable to count the frequency of "characters"
		 * The implementation of std::unordered_map is used a hashtable
		 * 
		 * @see: https://en.cppreference.com/w/cpp/container/unordered_map
		 * 
		 */
		std::unordered_map<CharType, size_t> map_freq_table;

		for (; begin != end; ++begin)
			++map_freq_table[static_cast<CharType>(*begin)];

		return detail::freq_table_type<CharType>(
			map_freq_table.begin(), map_freq_table.end()
		);
	}

	/**
	 * @brief Builds a Huffman tree from a frequency table
	 * 
	 * @tparam CharType  The type of "character" that will be encoded/decoded
	 * @param freq_table A constant reference for a type frequency table
	 * 					 (std::vector<std::pair<CharType, size_t>>)(aliased by 
	 * 					 detail::freq_table_type)
	 * @return 
	 */
	template <typename CharType = char8_t>
	constexpr auto BuildHuffmanTree(const auto& freq_table)
	{
		using node_type = HuffmanTreeNode<CharType>;
		using comp = typename node_type::Comparator;
		constexpr auto null_value = static_cast<CharType>(0);

		/**
		 * 
		 * @brief Here i used a min heap, a priority queue that will be used
		 * To generate the tree, we have the O(log n) in insertion and removal.
		 * 
		 * Note that STL already has a priority queue structure, I am using
		 * not only per issue of performance, but also by portability and compatibility.
		 * 
		 */
		std::priority_queue<node_type*, std::vector<node_type *>, comp> pqueue;

		// inserts all Freq_table pairs in the priority queue
		for (const auto& elem : freq_table)
			pqueue.push(new node_type(elem.first, elem.second));

		node_type* root  = nullptr;
		node_type* left  = nullptr;
		node_type* right = nullptr;

		/** 
		 * @brief 
		 * Here logic is as follows:
		 * 1 - While the queue has more than 1 element, we took the two of us with
		 *	   minors frequencies in the queue.
		 * 2 - We create a new node in memory as the sum of the frequencies of
		 *     nodes smaller than we extract.
		 * 3 - We put this new node generated inside the queue.
		 * 4 - Back to step 1.
		 */
		while (pqueue.size() > 1)
		{
			left = pqueue.top();
			pqueue.pop();

			right = pqueue.top();
			pqueue.pop();

			root = new node_type(null_value, left->frequency + right->frequency, 
								 left, right);
			pqueue.push(root);
		}

		root = pqueue.top();
		pqueue.pop();

		auto num_nodes = detail::GetNumberOfNodes(root);
		return HuffmanTree<CharType>(root, std::get<0>(num_nodes), std::get<1>(num_nodes));
	}

	/**
	 * @brief Build the code table, which is a hashtable of type pairs
	 * 		  ("character", "code word"), more precisely (Chartype, code_word_t).
	 * 
	 * @tparam CharType The type of "character" that will be encoded/decoded.
	 * @param huffman_tree An immutable reference for a Huffman tree (HuffmanTree).
	 * @return The code table (code_table_type).
	 */
	template <typename CharType>
	auto BuildCodeTable(const auto& huffman_tree)
	{
		using tree_type = std::remove_cvref_t<decltype(huffman_tree)>;
		using node_type = typename tree_type::node_type;
		using detail::code_word_type;
		using detail::code_table_type;

		code_table_type<CharType> code_table;

		code_word_type temp;
		code_word_type code_word;

		node_type* node 	   = nullptr;
		node_type* left_child  = nullptr; 
		node_type* right_child = nullptr;

		/** 
		 * @brief Here we will be using a queue with a pointer at the beginning and in the end
		 * For the purpose of saving pairs of type values ​​(node, code word).
		 * 
		 * This will be needed to label the sides of the Huffman tree
		 * or with '0' (left side) or with '1' right side.
		 * 
		 * The reason for using the queue was because of performance, use recursion
		 * to solve the problem takes a larger 50x processing time out the absurd 
		 * memory spending.
		 * 
		 */
		std::deque<std::pair<node_type*, code_word_type>> dqueue;
		dqueue.emplace_back(huffman_tree.GetRootNode(), code_word_type{});

		while (!dqueue.empty())
		{
			node = dqueue.front().first;
			code_word = dqueue.front().second;

			dqueue.pop_front();

			left_child = node->left;
			right_child = node->right;

			/**
			 * @brief As already said before, the Huffman tree is Full Binary Tree type,
			 * which means that the node can have only 0 or 2 children.
			 *
			 * Here, we could check if the 2 children are present, doing:
			 * if (left_child && right_child)
			 *
			 * but we can simplify this by doing only one verification in place of
			 * Two to save processing.
			 *
			 * You can verify whether the left child or the right is present in this
			 * If I have chosen the left if the left child is not present, either
			 * Say we arrived on a leaf node.
			 *
			 */
			if (left_child)
			{
				temp = code_word;
				
				// labeling '0' for the left child
				code_word.push_back(false);
				dqueue.emplace_back(left_child, std::move(code_word));

				// labeling '1' for the right child
				temp.push_back(true);
				dqueue.emplace_back(right_child, std::move(temp));
			}
			else
			{
				/**
				 * @brief The leaf node contains the "character", now we pushed both
				 * The "character" and the code word into the hashtable.
				 */
				code_table.emplace(node->ch, std::move(code_word));
			}
		}

		return code_table;
	}

	/**
	 * @brief Writes the Huffman header in output iterator,
	 *		  that is nothing more than the information needed to unzip
	 *		  The file later.
	 * 
	 * @tparam CharType The type of "character" that will be encoded/decoded
	 * @tparam OutputIter The type of output iterator
	 * @param freq_table An immutable reference for the frequency table 
	 * 					 (detail::freq_table_type)
	 * @param out The output iterator
	 */
	template <typename CharType, typename OutputIter>
	constexpr inline auto WriteHuffmanHeader(const auto& freq_table, OutputIter out)
	{
		static_assert(std::same_as<std::remove_cvref_t<decltype(freq_table)>, 
					  detail::freq_table_type<CharType>>);

		detail::WriteObjInMem(HuffmanSignature, out);
		detail::WriteObjInMem(freq_table.size(), out);
		detail::WriteObjInMem(freq_table[0], out, freq_table.size());
	}

	/**
	 * @brief Read the huffman header from the compressed file
	 * 
	 * @tparam CharType The type of "character" that will be encoded/decoded
	 * @tparam InputIter The type of input iterator
	 * @param begin An "iterator" for the beginning of the container/memory
	 * @param end   An "iterator" for the end of the container/memory
	 * @return The frequency table extracted from the compressed file header
	 */
	template <typename CharType, typename InputIter>
	auto ReadHuffmanHeader(InputIter& input)
	{
		uint32_t signature = {};
		detail::ReadObjInMem(input, signature);
		input += sizeof(signature);

		detail::Ensure(signature == HuffmanSignature, 
					  "[x] invalid or corrupted huffman header file");

		size_t count = 0;
		detail::ReadObjInMem(input, count);
		input += sizeof(count);

		detail::freq_table_type<CharType> frequency_table(count);

		detail::ReadObjInMem(input, frequency_table[0], frequency_table.size());
		input += count * sizeof(decltype(frequency_table[0]));

		return frequency_table;
	}

	/**
	 * @brief Compress a file or region from memory using the huffman algorithm.
	 * 
	 * @tparam CharType   The type of "character" that will be encoded/decoded
	 * @tparam InputIter  The type of input iterator
	 * @tparam OutputIter The type of output iterator
	 * @param begin An "iterator" for the beginning of the container/memory
	 * @param end   An "iterator" for the end of the container/memory
	 * @param out   An output "iterator" in which the compressed data will be written
	 */
	template <typename CharType, std::input_iterator InputIter, typename OutputIter>
	void HuffmanCompress(InputIter begin, InputIter end, OutputIter out)
	{
		using namespace detail;

		// Build Frequency Table
		auto frequency_table = BuildFrequencyTable<CharType>(begin, end);
		printf("[+] Frequency Table Size: %zu\n", frequency_table.size());

		// Build huffman tree
		auto huff_tree = BuildHuffmanTree<CharType>(frequency_table);
		printf("[+] Tree Nodes Count: %zu\n", huff_tree.GetLength());
		printf("[+] Tree Size: %zu bytes\n", huff_tree.GetSizeInBytes());

		// Build Code Table Lookup
		auto code_table = BuildCodeTable<CharType>(huff_tree);
		printf("[+] Code Table Size: %zu\n", code_table.size());

		code_word_type result;
		
		printf("[+] Getting all code words corresponding to the characters of the text\n");

		for (; begin != end; ++begin)
		{
			const code_word_type& word = code_table[static_cast<CharType>(*begin)];
			result.insert(result.end(), word.begin(), word.end());
		}

		printf("[+] Writing Huffman Header (Frequency Table)\n");
		WriteHuffmanHeader<CharType>(frequency_table, out);

		printf("[+] Writing Code Words\n");
		WriteCodewordTo<CharType>(std::move(result), out);
	}

	/**
	 * @brief Unzip the code word to the original data, and writes them in "out"
	 * 
	 * @tparam CodewordIterator An iterator of a code word
	 * @tparam HuffmanTreeModel Huffman Tree Type
	 * @tparam OutputIter An output iterator
	 * @param begin An "iterator" for the beginning of the code word
	 * @param end An "iterator" for the end of the code word
	 * @param huff_tree A huffman tree
	 * @param out An output iterator to write original unzipped data
	 */
	template <typename CodewordIterator,
			  typename HuffmanTreeModel,
			  typename OutputIter>
	constexpr void DecodeCodeword(CodewordIterator begin, CodewordIterator end,
								  const HuffmanTreeModel& huff_tree, OutputIter out)
	{
		using HuffmanTreeNodeType = typename HuffmanTreeModel::node_type;
		using IterValueType = typename detail::iter_value_type<CodewordIterator>::type;
		
		HuffmanTreeNodeType* root = huff_tree.GetRootNode();
		HuffmanTreeNodeType* node = root;

		for (auto iter = begin; iter != end; ++iter)
		{
			if (*iter == static_cast<IterValueType>(1))
				node = node->right;
			else
				node = node->left;

			if (node->IsLeafNode())
			{
				*out++ = node->ch;
				node = root;
			}
		}
	}

	/**
	 * @brief Unzip a file/region of memory that was compressed with the 
	 * 		  huffman algorithm.
	 * 
	 * @tparam CharType   The type of "character" that will be encoded/decoded
	 * @tparam InputIter  The type of input iterator
	 * @tparam OutputIter The type of output iterator
	 * @param begin An iterator for the beginning of the container/memory
	 * @param end   An iterator for the end of the container/memory
	 * @param out   An output iterator in which compressed data will be written
	 */
	template <typename CharType, typename InputIter, typename OutputIter>
	void HuffmanDecompress(InputIter begin, InputIter end, OutputIter out)
	{
		using namespace detail;
		
		printf("[+] Reading Huffman Reader (Frequency Table)\n");
		auto freq_table = ReadHuffmanHeader<CharType>(begin);

		printf("[+] Building Huffman Tree\n");
		auto huff_tree = BuildHuffmanTree<CharType>(freq_table);

		printf("[+] Huffman Tree Nodes: %zu\n", huff_tree.GetLength());
		printf("[+] Huffman Tree Size: %zu bytes\n", huff_tree.GetSizeInBytes());

		if (freq_table.size() == 1)
		{
			size_t count = freq_table[0].second;
			std::fill_n(out, count, freq_table[0].first);
			return;
		}

		printf("[+] Transforming the entry into a vector of bits (CodeWord)\n");
		auto codeword = ToCodeword<CharType>(begin, end);

		printf("[+] Uncompressing vector bits for the original data\n");
		DecodeCodeword(codeword.begin(), codeword.end(), huff_tree, out); 
	}
}

#endif
