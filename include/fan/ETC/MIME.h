#pragma once

#include _WITCH_PATH(WITCH.h)
#include _WITCH_PATH(MAP/MAP.h)
#include _WITCH_PATH(STR/STR.h)

enum{
	MIME_error_e,
	MIME_unknown_e,
	MIME_aac_e,
	MIME_abw_e,
	MIME_arc_e,
	MIME_avi_e,
	MIME_azw_e,
	MIME_bin_e,
	MIME_bmp_e,
	MIME_bz_e,
	MIME_bz2_e,
	MIME_csh_e,
	MIME_css_e,
	MIME_csv_e,
	MIME_doc_e,
	MIME_docx_e,
	MIME_eot_e,
	MIME_epub_e,
	MIME_gz_e,
	MIME_gif_e,
	MIME_html_e,
	MIME_ico_e,
	MIME_ics_e,
	MIME_jar_e,
	MIME_jpeg_e,
	MIME_js_e,
	MIME_json_e,
	MIME_jsonld_e,
	MIME_midi_e,
	MIME_mjs_e,
	MIME_mp3_e,
	MIME_mpeg_e,
	MIME_mpkg_e,
	MIME_odp_e,
	MIME_ods_e,
	MIME_odt_e,
	MIME_oga_e,
	MIME_ogv_e,
	MIME_ogx_e,
	MIME_opus_e,
	MIME_otf_e,
	MIME_png_e,
	MIME_pdf_e,
	MIME_php_e,
	MIME_ppt_e,
	MIME_pptx_e,
	MIME_rar_e,
	MIME_rtf_e,
	MIME_sh_e,
	MIME_svg_e,
	MIME_swf_e,
	MIME_tar_e,
	MIME_tiff_e,
	MIME_ts_e,
	MIME_ttf_e,
	MIME_txt_e,
	MIME_vsd_e,
	MIME_wav_e,
	MIME_weba_e,
	MIME_webm_e,
	MIME_webp_e,
	MIME_woff_e,
	MIME_woff2_e,
	MIME_xhtml_e,
	MIME_xls_e,
	MIME_xlsx_e,
	MIME_xml_e,
	MIME_xul_e,
	MIME_zip_e,
	MIME_3gp_e,
	MIME_3g2_e,
	MIME_7z_e,
	MIME_flac_e,
	MIME_mp4_e,
	MIME_mkv_e
};

MAP_t _MIME_mapname;
void _MIME_mapname_in(const char *in, uint32_t out){
	MAP_in_cstr(&_MIME_mapname, in, &out, sizeof(uint32_t));
}
#define d(in_m, out_m) \
	_MIME_mapname_in(in_m, out_m);
PRE{
	MAP_open(&_MIME_mapname);
	d("aac", MIME_aac_e)
	d("abw", MIME_abw_e)
	d("arc", MIME_arc_e)
	d("avi", MIME_avi_e)
	d("azw", MIME_azw_e)
	d("bin", MIME_bin_e)
	d("bmp", MIME_bmp_e)
	d("bz", MIME_bz_e)
	d("bz2", MIME_bz2_e)
	d("csh", MIME_csh_e)
	d("css", MIME_css_e)
	d("csv", MIME_csv_e)
	d("doc", MIME_doc_e)
	d("docx", MIME_docx_e)
	d("eot", MIME_eot_e)
	d("epub", MIME_epub_e)
	d("gz", MIME_gz_e)
	d("gif", MIME_gif_e)
	d("htm", MIME_html_e)
	d("html", MIME_html_e)
	d("ico", MIME_ico_e)
	d("ics", MIME_ics_e)
	d("jar", MIME_jar_e)
	d("jpg", MIME_jpeg_e)
	d("jpeg", MIME_jpeg_e)
	d("js", MIME_js_e)
	d("json", MIME_json_e)
	d("jsonld", MIME_jsonld_e)
	d("mid", MIME_midi_e)
	d("midi", MIME_midi_e)
	d("mjs", MIME_mjs_e)
	d("mp3", MIME_mp3_e)
	d("mpeg", MIME_mpeg_e)
	d("mpkg", MIME_mpkg_e)
	d("odp", MIME_odp_e)
	d("ods", MIME_ods_e)
	d("odt", MIME_odt_e)
	d("oga", MIME_oga_e)
	d("ogv", MIME_ogv_e)
	d("ogx", MIME_ogx_e)
	d("opus", MIME_opus_e)
	d("otf", MIME_otf_e)
	d("png", MIME_png_e)
	d("pdf", MIME_pdf_e)
	d("php", MIME_php_e)
	d("ppt", MIME_ppt_e)
	d("pptx", MIME_pptx_e)
	d("rar", MIME_rar_e)
	d("rtf", MIME_rtf_e)
	d("sh", MIME_sh_e)
	d("svg", MIME_svg_e)
	d("swf", MIME_swf_e)
	d("tar", MIME_tar_e)
	d("tif", MIME_tiff_e)
	d("tiff", MIME_tiff_e)
	d("ts", MIME_ts_e)
	d("ttf", MIME_ttf_e)
	d("txt", MIME_txt_e)
	d("vsd", MIME_vsd_e)
	d("wav", MIME_wav_e)
	d("weba", MIME_weba_e)
	d("webm", MIME_webm_e)
	d("webp", MIME_webp_e)
	d("woff", MIME_woff_e)
	d("woff2", MIME_woff2_e)
	d("xhtml", MIME_xhtml_e)
	d("xls", MIME_xls_e)
	d("xlsx", MIME_xlsx_e)
	d("xml", MIME_xml_e)
	d("xul", MIME_xul_e)
	d("zip", MIME_zip_e)
	d("3gp", MIME_3gp_e)
	d("3g2", MIME_3g2_e)
	d("7z", MIME_7z_e)
	d("flac", MIME_flac_e)
	d("mp4", MIME_mp4_e)
	d("mkv", MIME_mkv_e)
}
#undef d

uint32_t MIME_namen(const uint8_t *name, uintptr_t length){
	if(name == 0){
		return MIME_error_e;
	}
	const uint8_t *begin = name + length - 1;
	uintptr_t w = STR_nchri(begin, length, '.', -1);
	if(w == -1){
		return MIME_unknown_e;
	}
	MAP_out_t out = MAP_out(&_MIME_mapname, (begin - w) + 1, w);
	if(out.length == 0){
		return MIME_unknown_e;
	}
	return *(uint32_t *)(out.data);
}

uint32_t MIME_name(const uint8_t *name){
	return MIME_namen(name, MEM_cstreu(name));
}
