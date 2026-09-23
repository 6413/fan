import std;
import fan;

void print_line(std::string& out, const std::string& path, std::size_t line_num, std::string_view line, std::pair<std::size_t, std::size_t> m) {
	auto trim = line.find_first_not_of(" \t");
	std::size_t off = trim == std::string_view::npos ? 0 : trim;
	line.remove_prefix(off);
	m.first -= off;
	out += fan::paint(fan::colors::purple * 1.3, path);
	out += ":";
	out += fan::paint(fan::colors::green / 1.5, std::to_string(line_num));
	out += ": ";
	out += line.substr(0, m.first);
	out += fan::paint(fan::colors::red, std::string(line.substr(m.first, m.second)));
	out += line.substr(m.first + m.second);
	out += "\n";
}

void print_matches(const std::string_view search_path, const std::string_view needle) {
	bool literal = needle.find('*') == std::string_view::npos;
	std::string out;
	fan::io::iterate_files_recursive(search_path,
			[&](const std::filesystem::path& full, const std::filesystem::path& rel) {
				std::string data;
				if (fan::io::file::read(full.generic_string(), &data)) {
					return;
				}
				std::string path = full.generic_string();
				if (literal) {
					std::size_t pos = 0, line_num = 1, line_start = 0;
					while ((pos = data.find(needle, pos)) != std::string::npos) {
						while (line_start <= pos) {
							std::size_t nl = data.find('\n', line_start);
							if (nl == std::string::npos || nl >= pos) {
								break;
							}
							line_start = nl + 1;
							++line_num;
						}
						std::size_t line_end = data.find('\n', pos);
						if (line_end == std::string::npos) {
							line_end = data.size();
						}
						std::string_view line(data.data() + line_start, line_end - line_start);
						print_line(out, path, line_num, line, {pos - line_start, needle.size()});
						pos = line_end;
					}
					return;
				}
				std::size_t line_num = 1, start = 0;
				while (start <= data.size()) {
					std::size_t end = data.find('\n', start);
					std::string_view line(data.data() + start, (end == std::string::npos ? data.size() : end) - start);
					if (auto m = fan::wildcard_find(line, needle)) {
						print_line(out, path, line_num, line, *m);
					}
					if (end == std::string::npos) {
						break;
					}
					start = end + 1;
					++line_num;
				}
		});
	fan::printr(out);
}

int main(int argc, char** argv) {
	fan::args_t args(argc, argv);
	if (args.size() < 2) {
		fan::print("usage: grep <pattern>");
		fan::print("       grep <path> <pattern>  ('*' matches anything)");
		return 1;
	}
	print_matches(args.size() >= 3 ? args[1] : "./", args.size() >= 3 ? args[2] : args[1]);
}
