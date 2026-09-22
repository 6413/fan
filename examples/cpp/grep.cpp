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
					fan::print(
						fan::paint(fan::colors::purple * 1.3, path) + ":"  +
						fan::paint(fan::colors::green / 1.5, line)  + ": " +
						str.substr(0, found) + fan::paint(fan::colors::red, needle) + str.substr(found + needle.size())
					);
				}
		});
}

int main(int argc, char** argv) {
	fan::args_t args(argc, argv);
	if (args.size() < 2) {
		fan::print("usage: grep <pattern>");
		fan::print("       grep <path> <pattern>");
		return 1;
	}
	print_matches(args.size() >= 3 ? args[1] : "./", args.size() >= 3 ? args[2] : args[1]);
}
