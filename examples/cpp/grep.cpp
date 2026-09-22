import std;
import fan;

void print_matches(const std::string_view search_path, const std::string_view needle) {
	fan::io::iterate_files_recursive(search_path,
			[&](const std::filesystem::path& full, const std::filesystem::path& rel) {
				std::ifstream file(full, std::ios::binary);
				std::string line;
				std::ostringstream oss;
				oss << file.rdbuf();
				std::string path = full.generic_string();
				if (auto [line, str] = fan::find_in_file(path, needle); str.size()) {
					str.erase(0, str.find_first_not_of(" \t"));

					auto found = str.find(needle);
					fan::printf("{}:{}: {}",
						fan::paint(fan::colors::purple * 1.3, path),
						fan::paint(fan::colors::green / 1.5, line),
						str.substr(0, found) + fan::paint(fan::colors::red, needle) + str.substr(found + needle.size())
					);
				}
		});
}

int main(int argc, char** argv) {
	fan::args_t fargs(argc, argv);
	auto& args = fargs.get();
	// we dont want program name ty
	args.erase(args.begin());
	print_matches(argc >= 3 ? args[0] : "./", argc >= 3 ? args[1] : args[0]);
}
