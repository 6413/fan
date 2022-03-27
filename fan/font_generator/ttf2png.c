/*
ttf2png - True Type Font to PNG converter
Copyright (c) 2004-2021 Mikko Rasa, Mikkosoft Productions

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <getopt.h>
#include <png.h>
#include <ft2build.h>
#include FT_FREETYPE_H

typedef struct sImage
{
	unsigned w, h;
	unsigned char *data;
	unsigned border;
} Image;

typedef struct sGlyph
{
	unsigned index;
	unsigned code;
	Image image;
	unsigned x, y;
	int offset_x;
	int offset_y;
	int advance;
} Glyph;

typedef struct sKerning
{
	Glyph *left_glyph;
	Glyph *right_glyph;
	int distance;
} Kerning;

typedef struct sFont
{
	unsigned size;
	int ascent;
	int descent;
	unsigned n_glyphs;
	Glyph *glyphs;
	unsigned n_kerning;
	Kerning *kerning;
	Image image;
} Font;

typedef struct sRange
{
	unsigned first;
	unsigned last;
} Range;

typedef unsigned char bool;

void usage(void);
int convert_numeric_option(char, int);
void convert_code_point_range(char, Range *);
int str_to_code_point(const char *, char **);
void convert_size(char, unsigned *, unsigned *);
void sort_and_compact_ranges(Range *, unsigned *);
int range_cmp(const void *, const void *);
unsigned round_to_pot(unsigned);
int init_image(Image *, size_t, size_t);
int init_font(Font *, FT_Face, const Range *, unsigned, bool, unsigned, unsigned);
int init_glyphs(Font *, FT_Face, const Range *, bool, unsigned, unsigned);
int copy_bitmap(const FT_Bitmap *, Image *);
unsigned sqrti(unsigned);
unsigned find_distance_to_edge(const Image *, int, int, unsigned);
int create_distance_field(const FT_Bitmap *, Image *, unsigned, unsigned);
int render_grid(Font *, unsigned, unsigned, unsigned, bool);
int render_packed(Font *, unsigned, unsigned);
int save_defs(const char *, const Font *);
int save_png(const char *, const Image *, bool, bool, bool);

char verbose = 0;

int main(int argc, char **argv)
{
	char *fn;
	Range *ranges = NULL;
	unsigned n_ranges = 0;
	unsigned size = 10;
	unsigned cpl = 0;
	unsigned cellw = 0;
	unsigned cellh = 0;
	bool autohinter = 0;
	bool seq = 0;
	bool alpha = 0;
	bool invert = 0;
	bool pack = 0;
	unsigned margin = 0;
	unsigned padding = 1;
	bool npot = 0;
	unsigned distfield = 0;
	unsigned border = 0;

	FT_Library freetype;
	FT_Face face;

	int err;
	int i;

	char *out_fn = "font.png";
	char *def_fn = NULL;

	Font font;

	if(argc<2)
	{
		usage();
		return 1;
	}

	while((i = getopt(argc, argv, "r:s:l:c:o:atvh?ed:pim:n:gf:b:")) != -1)
	{
		switch(i)
		{
		case 'r':
			ranges = (Range *)realloc(ranges, (++n_ranges)*sizeof(Range));
			convert_code_point_range('r', &ranges[n_ranges-1]);
			break;
		case 's':
			size = convert_numeric_option('s', 1);
			break;
		case 'l':
			cpl = convert_numeric_option('l', 1);
			break;
		case 'c':
			convert_size('c', &cellw, &cellh);
			break;
		case 'o':
			out_fn = optarg;
			break;
		case 'a':
			autohinter = 1;
			break;
		case 't':
			alpha = 1;
			break;
		case 'v':
			++verbose;
			break;
		case 'h':
		case '?':
			usage();
			return 0;
		case 'e':
			seq = 1;
			break;
		case 'd':
			def_fn = optarg;
			break;
		case 'p':
			pack = 1;
			break;
		case 'i':
			invert = 1;
			break;
		case 'm':
			margin = convert_numeric_option('m', 0);
			break;
		case 'n':
			padding = convert_numeric_option('n', 0);
			break;
		case 'g':
			npot = 1;
			break;
		case 'f':
			distfield = convert_numeric_option('f', 1);
			break;
		case 'b':
			border = convert_numeric_option('b', 1);
			break;
		}
	}
	if(!strcmp(out_fn, "-"))
		verbose = 0;

	if(optind!=argc-1)
	{
		usage();
		return 1;
	}

	fn = argv[optind];

	err = FT_Init_FreeType(&freetype);
	if(err)
	{
		fprintf(stderr, "Couldn't initialize FreeType library\n");
		return 1;
	}

	err = FT_New_Face(freetype, fn, 0, &face);
	if(err)
	{
		fprintf(stderr, "Couldn't load font file\n");
		if(err==FT_Err_Unknown_File_Format)
			fprintf(stderr, "Unknown file format\n");
		return 1;
	}

	if(verbose)
	{
		const char *name = FT_Get_Postscript_Name(face);
		printf("Font name: %s\n", name);
		printf("Glyphs:    %ld\n", face->num_glyphs);
	}

	font.size = size;
	if(distfield)
	{
		if(!border)
			border = sqrti(font.size);
		size *= distfield;
	}

	err = FT_Set_Pixel_Sizes(face, 0, size);
	if(err)
	{
		fprintf(stderr, "Couldn't set size\n");
		return 1;
	}

	if(!n_ranges)
	{
		ranges = malloc(sizeof(Range));
		ranges[0].first = 0;
		ranges[0].last = 255;
		n_ranges = 1;
	}
	else
		sort_and_compact_ranges(ranges, &n_ranges);

	err = init_font(&font, face, ranges, n_ranges, autohinter, distfield, border);
	if(err)
		return 1;

	if(!font.n_glyphs)
	{
		fprintf(stderr, "No glyphs found in the requested range\n");
		return 1;
	}

	if(pack)
		err = render_packed(&font, margin, padding);
	else
		err = render_grid(&font, cellw, cellh, cpl, seq);
	if(err)
		return 1;

	err = save_png(out_fn, &font.image, (alpha && !distfield), (invert || distfield), npot);
	if(err)
		return 1;

	if(def_fn)
		save_defs(def_fn, &font);

	for(i=0; (unsigned)i<font.n_glyphs; ++i)
		free(font.glyphs[i].image.data);
	free(font.glyphs);
	free(font.kerning);
	free(font.image.data);
	free(ranges);

	FT_Done_Face(face);
	FT_Done_FreeType(freetype);

	return 0;
}

void usage(void)
{
	printf("ttf2png 2.0 - True Type Font to PNG converter\n"
		"Copyright (c) 2004-2021  Mikko Rasa, Mikkosoft Productions\n"
		"Distributed under the GNU General Public License\n\n");

	printf("Usage: ttf2png [options] <TTF file>\n\n");

	printf("Accepted options (default values in [brackets])\n"
		"  -r  Range of code points to convert [0,255]\n"
		"  -s  Font size to use, in pixels [10]\n"
		"  -l  Number of glyphs to put in one line [auto]\n"
		"  -c  Glyph cell size, in pixels (grid mode only) [auto]\n"
		"  -o  Output file name (or - for stdout) [font.png]\n");
	printf("  -a  Force autohinter\n"
		"  -t  Render glyphs to alpha channel\n"
		"  -i  Invert colors of the glyphs\n"
		"  -v  Increase the level of verbosity\n"
		"  -e  Use cells in sequence, without gaps (grid mode only)\n"
		"  -p  Pack the glyphs tightly instead of in a grid\n"
		"  -m  Margin around image edges (packed mode only) [0]\n"
		"  -n  Padding between glyphs (packed mode only) [1]\n"
		"  -g  Allow non-power-of-two result\n");
	printf("  -f  Create a distance field texture\n"
		"  -b  Specify distance field border zone width\n"
		"  -d  File name for writing glyph definitions\n"
		"  -h  Print this message\n");
}

int convert_numeric_option(char opt, int min_value)
{
	int value;
	char *ptr;

	value = strtol(optarg, &ptr, 0);
	if(value<min_value || *ptr)
	{
		printf("Invalid option argument in -%c %s\n", opt, optarg);
		exit(1);
	}

	return value;
}

void convert_code_point_range(char opt, Range *range)
{
	int value;
	char *ptr;

	if(!strcmp(optarg, "all"))
	{
		range->first = 0;
		range->last = 0x10FFFF;
		return;
	}

	value = str_to_code_point(optarg, &ptr);
	if(value>=0 && *ptr==',')
	{
		range->first = value;
		value = str_to_code_point(ptr+1, &ptr);
		if(value>=(int)range->first && !*ptr)
		{
			range->last = value;
			return;
		}
	}

	printf("Invalid option argument in -%c %s\n", opt, optarg);
	exit(1);
}

int str_to_code_point(const char *nptr, char **endptr)
{
	if(nptr[0]=='U' && nptr[1]=='+')
		return strtoul(nptr+2, endptr, 16);
	else if(nptr[0]&0x80)
	{
		unsigned bytes;
		unsigned code;
		unsigned i;

		if(endptr)
			*endptr = (char *)nptr;

		for(bytes=1; (bytes<4 && (nptr[0]&(0x80>>bytes))); ++bytes)
			if((nptr[bytes]&0xC0)!=0x80)
				return -1;
		if(bytes<2)
			return -1;

		code = nptr[0]&(0x3F>>bytes);
		for(i=1; i<bytes; ++i)
			code = (code<<6)|(nptr[i]&0x3F);

		if(endptr)
			*endptr = (char *)nptr+bytes;

		return code;
	}
	else if(isdigit(nptr[0]))
		return strtoul(nptr, endptr, 0);
	else
	{
		if(endptr)
			*endptr = (char *)nptr+1;
		return *nptr;
	}
}

void convert_size(char opt, unsigned *width, unsigned *height)
{
	int value;
	char *ptr;

	if(!strcmp(optarg, "auto"))
	{
		*width = 0;
		*height = 0;
		return;
	}
	else if(!strcmp(optarg, "autorect"))
	{
		*width = 0;
		*height = 1;
		return;
	}

	value = strtol(optarg, &ptr, 0);
	if(value>0)
	{
		*width = value;
		if(*ptr=='x')
		{
			value = strtol(ptr+1, &ptr, 0);
			if(value>0 && !*ptr)
			{
				*height = value;
				return;
			}
		}
		else if(!*ptr)
		{
			*height = *width;
			return;
		}
	}

	printf("Invalid option argument in -%c %s\n", opt, optarg);
	exit(1);
}

void sort_and_compact_ranges(Range *ranges, unsigned *n_ranges)
{
	unsigned i, j;

	if(!*n_ranges)
		return;

	qsort(ranges, *n_ranges, sizeof(Range), &range_cmp);
	for(i=0, j=1; j<*n_ranges; ++j)
	{
		if(ranges[i].last+1>=ranges[j].first)
		{
			if(ranges[j].last>ranges[i].last)
				ranges[i].last = ranges[j].last;
		}
		else
		{
			++i;
			if(i!=j)
				ranges[i] = ranges[j];
		}
	}

	*n_ranges = i+1;
}

int range_cmp(const void *p1, const void *p2)
{
	const Range *r1 = (const Range *)p1;
	const Range *r2 = (const Range *)p2;
	if(r1->first!=r2->first)
		return (r1->first<r2->first ? -1 : 1);
	else if(r1->last!=r2->last)
		return (r1->last<r2->last ? -1 : 1);
	else
		return 0;
}

unsigned round_to_pot(unsigned n)
{
	n -= 1;
	n |= n>>1;
	n |= n>>2;
	n |= n>>4;
	n |= n>>8;
	n |= n>>16;

	return n+1;
}

int init_image(Image *image, size_t w, size_t h)
{
	size_t s;

	image->w = w;
	image->h = h;
	image->data = NULL;
	image->border = 0;

	if(!image->w || !image->h)
		return 0;

	s = w*h;
	if(s/h!=w)
	{
		fprintf(stderr, "Cannot allocate memory for a %dx%d image\n", image->w, image->h);
		return -1;
	}

	image->data = malloc(s);
	if(!image->data)
	{
		fprintf(stderr, "Cannot allocate memory for a %dx%d image\n", image->w, image->h);
		return -1;
	}

	return 0;
}

int init_font(Font *font, FT_Face face, const Range *ranges, unsigned n_ranges, bool autohinter, unsigned distfield, unsigned border)
{
	unsigned i, j;
	unsigned size = 0;
	int scale = (distfield>0 ? distfield : 1);

	font->ascent = (face->size->metrics.ascender/scale+63)/64;
	font->descent = (face->size->metrics.descender/scale-63)/64;

	if(verbose>=1)
	{
		printf("Ascent:    %d\n", font->ascent);
		printf("Descent:   %d\n", font->descent);
	}

	font->n_glyphs = 0;
	font->glyphs = NULL;
	for(i=0; i<n_ranges; ++i)
		if(init_glyphs(font, face, &ranges[i], autohinter, distfield, border))
			return -1;

	if(verbose>=1)
		printf("Loaded %u glyphs\n", font->n_glyphs);

	font->n_kerning = 0;
	font->kerning = NULL;
	for(i=0; i<font->n_glyphs; ++i) for(j=0; j<font->n_glyphs; ++j)
		if(j!=i)
		{
			FT_Vector kerning;
			FT_Get_Kerning(face, font->glyphs[i].index, font->glyphs[j].index, FT_KERNING_DEFAULT, &kerning);

			/* FreeType documentation says that vertical kerning is practically
			never used, so we ignore it. */
			if(kerning.x)
			{
				Kerning *kern;

				if(font->n_kerning>=size)
				{
					size += 16;
					font->kerning = (Kerning *)realloc(font->kerning, size*sizeof(Kerning));
				}

				kern = &font->kerning[font->n_kerning++];
				kern->left_glyph = &font->glyphs[i];
				kern->right_glyph = &font->glyphs[j];
				kern->distance = (kerning.x/scale+32)/64;
			}
		}

	if(verbose>=1)
		printf("Loaded %d kerning pairs\n", font->n_kerning);

	return 0;
}

int init_glyphs(Font *font, FT_Face face, const Range *range, bool autohinter, unsigned distfield, unsigned border)
{
	unsigned i, j;
	unsigned size = font->n_glyphs;
	int scale = (distfield>0 ? distfield : 1);

	for(i=range->first; i<=range->last; ++i)
	{
		unsigned n;
		FT_Bitmap *bmp = &face->glyph->bitmap;
		int flags = 0;
		Glyph *glyph;

		n = FT_Get_Char_Index(face, i);
		if(!n)
			continue;

		if(autohinter)
			flags |= FT_LOAD_FORCE_AUTOHINT;
		FT_Load_Glyph(face, n, flags);
		FT_Render_Glyph(face->glyph, (distfield ? FT_RENDER_MODE_MONO : FT_RENDER_MODE_NORMAL));

		if(verbose>=2)
		{
			printf("  Code point U+%04X", i);
			if(i>=0x20 && i<0x7F)
				printf(" (%c)", i);
			else if(i>=0xA0 && i<=0x10FFFF)
			{
				char utf8[5];
				unsigned bytes;

				for(bytes=2; i>>(1+bytes*5); ++bytes) ;
				for(j=0; j<bytes; ++j)
					utf8[j] = 0x80 | ((i>>((bytes-j-1)*6))&0x3F);
				utf8[0] |= 0xF0<<(4-bytes);
				utf8[j] = 0;

				printf(" (%s)", utf8);
			}
			printf(": glyph %u, size %dx%d\n", n, bmp->width/scale, bmp->rows/scale);
		}

		if(bmp->pixel_mode!=FT_PIXEL_MODE_GRAY && bmp->pixel_mode!=FT_PIXEL_MODE_MONO)
		{
			fprintf(stderr, "Warning: Glyph %u skipped, incompatible pixel mode\n", n);
			continue;
		}

		if(font->n_glyphs>=size)
		{
			size += 16;
			font->glyphs = (Glyph *)realloc(font->glyphs, size*sizeof(Glyph));
		}

		glyph = &font->glyphs[font->n_glyphs++];
		glyph->index = n;
		glyph->code = i;
		glyph->offset_x = (int)(face->glyph->bitmap_left+scale/2)/scale;
		glyph->offset_y = (int)(face->glyph->bitmap_top-bmp->rows+scale/2)/scale;
		glyph->advance = (int)(face->glyph->advance.x/scale+32)/64;

		/* Copy the glyph image since FreeType uses a global buffer, which would
		be overwritten by the next glyph.  Negative pitch means the scanlines
		start from the bottom. */
		if(distfield)
		{
			glyph->offset_x -= border;
			glyph->offset_y -= border;
			create_distance_field(bmp, &glyph->image, distfield, border);
		}
		else
		{
			if(copy_bitmap(bmp, &glyph->image))
				return -1;
		}
	}

	return 0;
}

int copy_bitmap(const FT_Bitmap *bmp, Image *image)
{
	unsigned x, y;
	unsigned char *src;
	unsigned char *dst;

	if(init_image(image, bmp->width, bmp->rows))
		return -1;
	if(!image->w || !image->h)
		return 0;

	if(bmp->pitch<0)
		src = bmp->buffer+(bmp->rows-1)*-bmp->pitch;
	else
		src = bmp->buffer;
	dst = image->data;

	for(y=0; y<bmp->rows; ++y)
	{
		if(bmp->pixel_mode==FT_PIXEL_MODE_MONO)
		{
			for(x=0; x<bmp->width; ++x)
				dst[x] = ((src[x/8]&(0x80>>(x%8))) ? 0xFF : 0x00);
		}
		else
		{
			for(x=0; x<bmp->width; ++x)
				dst[x] = src[x];
		}

		src += bmp->pitch;
		dst += image->w;
	}

	return 0;
}

unsigned sqrti(unsigned num)
{
	unsigned result = (num>0xFFFF ? 0xFFFF : 0x100);
	while(result && result*result>=result+num)
		result -= (result*result+result-num)/(result*2);

	return result;
}

unsigned find_distance_to_edge(const Image *image, int origin_x, int origin_y, unsigned range)
{
	unsigned i, j;
	int x, y;
	unsigned char origin_pixel = 0;
	unsigned closest = range*range;

	if(origin_x>=0 && (unsigned)origin_x<image->w && origin_y>=0 && (unsigned)origin_y<image->h)
		origin_pixel = image->data[origin_x+origin_y*image->w];

	x = origin_x-1;
	y = origin_y-1;
	for(i=1; (i<range && i*i<=closest); ++i, --x, --y) for(j=0; j<4; ++j)
	{
		unsigned k;
		int dx = (j==0 ? 1 : j==2 ? -1 : 0);
		int dy = (j==1 ? 1 : j==3 ? -1 : 0);

		for(k=0; k<i*2; ++k, x+=dx, y+=dy)
		{
			unsigned char pixel = 0;
			if(x>=0 && (unsigned)x<image->w && y>=0 && (unsigned)y<image->h)
				pixel = image->data[x+y*image->w];
				
			if((pixel^origin_pixel)&0x80)
			{
				unsigned d = 2*i*i + k*k - 2*k*i;
				if(d<closest)
					closest = d;
			}
		}
	}

	return sqrti(closest*0x3F01)/range;
}

int create_distance_field(const FT_Bitmap *bmp, Image *image, unsigned scale, unsigned margin)
{
	unsigned x, y;
	Image base_image;

	if(init_image(image, (bmp->width+scale-1)/scale+2*margin, (bmp->rows+scale-1)/scale+2*margin))
		return -1;
	if(!image->w || !image->h)
		return 0;

	if(copy_bitmap(bmp, &base_image))
		return -1;

	image->border = margin;
	for(y=0; y<image->h; ++y) for(x=0; x<image->w; ++x)
	{
		int bx = (x-margin)*scale+scale/2;
		int by = (y-margin)*scale+scale/2;
		unsigned char pixel = find_distance_to_edge(&base_image, bx, by, margin*scale);
		if(bx>=0 && (unsigned)bx<base_image.w && by>=0 && (unsigned)by<base_image.h)
			pixel |= base_image.data[bx+by*base_image.w]&0x80;
		if(!(pixel&0x80))
			pixel = 0x80-pixel;
		image->data[x+y*image->w] = pixel;
	}

	free(base_image.data);

	return 0;
}

int render_grid(Font *font, unsigned cellw, unsigned cellh, unsigned cpl, bool seq)
{
	unsigned i;
	int top = 0, bot = 0;
	unsigned first, n_cells;
	unsigned maxw = 0, maxh = 0;

	/* Find extremes of the glyph images. */
	for(i=0; i<font->n_glyphs; ++i)
	{
		int y;

		y = font->glyphs[i].offset_y+font->glyphs[i].image.h;
		if(y>top)
			top = y;
		if(font->glyphs[i].offset_y<bot)
			bot = font->glyphs[i].offset_y;
		if(font->glyphs[i].image.w>maxw)
			maxw = font->glyphs[i].image.w;
		if(font->glyphs[i].image.h>maxh)
			maxh = font->glyphs[i].image.h;
	}

	if(cellw==0)
	{
		/* Establish a large enough cell to hold all glyphs in the range. */
		int square = (cellh==cellw);
		cellw = maxw;
		cellh = top-bot;
		if(square)
		{
			if(cellh>cellw)
				cellw = cellh;
			else
				cellh = cellw;
		}
	}

	if(verbose>=1)
	{
		printf("Max size:  %u x %u\n", maxw, maxh);
		printf("Y range:   [%d %d]\n", bot, top);
		printf("Cell size: %u x %u\n", cellw, cellh);
		if(maxw>cellw || (unsigned)(top-bot)>cellh)
			fprintf(stderr, "Warning: character size exceeds cell size\n");
	}

	if(cpl==0)
	{
		/* Determine number of characters per line, trying to fit all the glyphs
		in a square image. */
		for(i=1;; i<<=1)
		{
			cpl = i/cellw;
			if(cpl>0 && font->n_glyphs/cpl*cellh<=cpl*cellw)
				break;
		}
	}

	first = font->glyphs[0].code;
	if(seq)
		n_cells = font->n_glyphs;
	else
	{
		first -= first%cpl;
		n_cells = font->glyphs[font->n_glyphs-1].code+1-first;
	}

	if(init_image(&font->image, cpl*cellw, (n_cells+cpl-1)/cpl*cellh))
		return -1;
	memset(font->image.data, 0, font->image.w*font->image.h);

	for(i=0; i<font->n_glyphs; ++i)
	{
		Glyph *glyph;
		unsigned ci, cx, cy;
		unsigned x, y;

		glyph = &font->glyphs[i];

		if(seq)
			ci = i;
		else
			ci = glyph->code-first;

		cx = (ci%cpl)*cellw;
		cy = (ci/cpl)*cellh;

		if(cellw>glyph->image.w)
			cx += (cellw-glyph->image.w)/2;
		cy += top-glyph->offset_y-glyph->image.h;

		glyph->x = cx;
		glyph->y = cy;

		for(y=0; y<glyph->image.h; ++y) for(x=0; x<glyph->image.w; ++x)
		{
			if(cx+x>=font->image.w || cy+y>=font->image.h)
				continue;
			font->image.data[cx+x+(cy+y)*font->image.w] = glyph->image.data[x+y*glyph->image.w];
		}
	}

	return 0;
}

int render_packed(Font *font, unsigned margin, unsigned padding)
{
	unsigned i;
	size_t area = 0;
	bool *used_glyphs;
	unsigned *used_pixels;
	unsigned cx = margin, cy;
	unsigned used_h = 0;

	/* Compute the total area occupied by glyphs and padding. */
	for(i=0; i<font->n_glyphs; ++i)
	{
		size_t a = area+(font->glyphs[i].image.w+padding)*(font->glyphs[i].image.h+padding);
		if(a<area)
		{
			fprintf(stderr, "Overflow in counting total glyph area\n");
			return -1;
		}
		area = a;
	}

	/* Find an image size that's approximately square. */
	for(font->image.w=1;; font->image.w<<=1)
	{
		if(font->image.w<=margin*2)
			continue;
		font->image.h = area/(font->image.w-margin*2)+margin*2;
		if(font->image.h<=font->image.w)
			break;
	}

	/* Add some extra space to accommodate packing imperfections. */
	font->image.h = font->image.h*3/2;

	/* Allocate arrays for storing the image and keeping track of used pixels and
	glyphs.  Since glyphs are rectangular and the image is filled starting from
	the top, it's enough to track the number of used pixels at the top of each
	column. */
	if(init_image(&font->image, font->image.w, font->image.h))
		return -1;
	memset(font->image.data, 0, font->image.w*font->image.h);
	used_pixels = (unsigned *)malloc(font->image.w*sizeof(unsigned));
	memset(used_pixels, 0, font->image.w*sizeof(unsigned));
	used_glyphs = (bool *)malloc(font->n_glyphs);
	memset(used_glyphs, 0, font->n_glyphs);

	for(cy=margin; cy+margin<font->image.h;)
	{
		unsigned w;
		unsigned x, y;
		Glyph *glyph = NULL;
		unsigned best_score = 0;
		unsigned target_h = 0;

		/* Find the leftmost free pixel on this row.  Also record the lowest
		extent of glyphs to the left of the free position. */
		for(; (cx+margin<font->image.w && used_pixels[cx]>cy); ++cx)
			if(used_pixels[cx]-cy-padding>target_h)
				target_h = used_pixels[cx]-cy-padding;

		if(cx+margin>=font->image.w)
		{
			cx = margin;
			++cy;
			continue;
		}

		/* Count the free pixel at this position. */
		for(w=0; (cx+w+margin<font->image.w && used_pixels[cx+w]<=cy); ++w) ;

		/* Find a suitable glyph to put here. */
		for(i=0; i<font->n_glyphs; ++i)
		{
			Glyph *g;

			g = &font->glyphs[i];
			if(!used_glyphs[i] && g->image.w<=w)
			{
				unsigned score;

				/* Prefer glyphs that would reach exactly as low as the ones left
				of here.  This aims to create a straight edge at the bottom for
				lining up further glyphs. */
				score = g->image.h+padding;
				if(g->image.h==target_h)
					score *= g->image.w;
				else
					score += g->image.w;

				if(score>best_score)
				{
					glyph = g;
					best_score = score;
				}
			}
		}

		if(!glyph)
		{
			cx += w;
			continue;
		}

		used_glyphs[glyph-font->glyphs] = 1;
		glyph->x = cx;
		glyph->y = cy;

		for(y=0; y<glyph->image.h; ++y) for(x=0; x<glyph->image.w; ++x)
		{
			if(cx+x>=font->image.w || cy+y>=font->image.h)
				continue;
			font->image.data[cx+x+(cy+y)*font->image.w] = glyph->image.data[x+y*glyph->image.w];
		}
		for(x=0; x<glyph->image.w+2*padding; ++x)
		{
			if(cx+x<padding || cx+x>=font->image.w+padding)
				continue;
			if(used_pixels[cx+x-padding]<cy+glyph->image.h+padding)
				used_pixels[cx+x-padding] = cy+glyph->image.h+padding;
		}

		if(cy+glyph->image.h+margin>used_h)
			used_h = cy+glyph->image.h+margin;
	}

	/* Trim the image to the actually used size, in case the original estimate
	was too pessimistic. */
	font->image.h = used_h;

	free(used_glyphs);
	free(used_pixels);

	return 0;
}

int save_defs(const char *fn, const Font *font)
{
	FILE *out;
	unsigned i;

	out = fopen(fn, "w");
	if(!out)
	{
		fprintf(stderr, "Couldn't open %s\n",fn);
		return -1;
	}

	fprintf(out, "# Image/font info:\n");
	fprintf(out, "# width height size ascent descent\n");
	fprintf(out, "font %d %d %d %d %d\n", font->image.w, font->image.h, font->size, font->ascent, font->descent);

	fprintf(out, "\n# Code point mapping:\n");
	fprintf(out, "# code index\n");
	for(i=0; i<font->n_glyphs; ++i)
	{
		const Glyph *g = &font->glyphs[i];
		fprintf(out, "code %u %u\n", g->code, g->index);
	}

	fprintf(out, "\n# Metrics info:\n");
	fprintf(out, "# index width height offset_x offset_y advance\n");
	for(i=0; i<font->n_glyphs; ++i)
	{
		const Glyph *g = &font->glyphs[i];
		int b = g->image.border;
		fprintf(out, "metrics %u %u %u %d %d %d\n", g->index, g->image.w-2*b, g->image.h-2*b, g->offset_x+b, g->offset_y+b, g->advance);
	}

	fprintf(out, "\n# Glyph info:\n");
	fprintf(out, "# index x y width height border\n");
	for(i=0; i<font->n_glyphs; ++i)
	{
		const Glyph *g = &font->glyphs[i];
		fprintf(out, "glyph %u %u %u %u %u %u\n", g->index, g->x, g->y, g->image.w, g->image.h, g->image.border);
	}

	fprintf(out, "\n# Kerning info:\n");
	fprintf(out, "# left right distance\n");
	for(i=0; i<font->n_kerning; ++i)
	{
		const Kerning *k = &font->kerning[i];
		fprintf(out, "kern %u %u %d\n", k->left_glyph->index, k->right_glyph->index, k->distance);
	}

	fclose(out);

	return 0;
}

int save_png(const char *fn, const Image *image, bool alpha, bool invert, bool npot)
{
	FILE *out;
	png_struct *pngs;
	png_info *pngi;
	unsigned w, h;
	png_byte *row;
	unsigned x, y;
	int color;
	unsigned flip_bits = (invert==alpha ? 0xFF : 0x00);
	unsigned char *src = image->data;

	if(!strcmp(fn, "-"))
		out = stdout;
	else
	{
		out = fopen(fn, "wb");
		if(!out)
		{
			fprintf(stderr, "Couldn't open %s\n",fn);
			return -1;
		}
	}

	pngs = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!pngs)
	{
		fprintf(stderr, "Error writing PNG file\n");
		return -1;
	}
	pngi = png_create_info_struct(pngs);
	if(!pngi)
	{
		png_destroy_write_struct(&pngs, NULL);
		fprintf(stderr, "Error writing PNG file\n");
		return -1;
	}

	w = (npot ? image->w : round_to_pot(image->w));
	h = (npot ? image->h : round_to_pot(image->h));
	color = (alpha ? PNG_COLOR_TYPE_GRAY_ALPHA : PNG_COLOR_TYPE_GRAY);
	png_set_IHDR(pngs, pngi, w, h, 8, color, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

	png_init_io(pngs, out);
	png_write_info(pngs, pngi);
	row = (png_byte *)malloc(w*(1+alpha));
	if(alpha)
	{
		for(x=0; x<w; ++x)
		{
			row[x*2] = 255;
			row[x*2+1] = flip_bits;
		}
		for(y=0; y<image->h; ++y)
		{
			for(x=0; x<image->w; ++x)
				row[x*2+1] = *src++^flip_bits;
			png_write_row(pngs, row);
		}

		for(x=0; x<w; ++x)
			row[x*2+1] = 0;
	}
	else
	{
		memset(row+image->w, flip_bits, w-image->w);
		for(y=0; y<image->h; ++y)
		{
			for(x=0; x<image->w; ++x)
				row[x] = *src++^flip_bits;
			png_write_row(pngs, row);
		}

		memset(row, flip_bits, w);
	}

	for(; y<h; ++y)
		png_write_row(pngs, row);

	png_write_end(pngs, pngi);
	png_destroy_write_struct(&pngs, &pngi);
	free(row);

	if(verbose)
		printf("Saved %dx%d PNG image to %s\n", w, h, fn);

	fclose(out);

	return 0;
}
