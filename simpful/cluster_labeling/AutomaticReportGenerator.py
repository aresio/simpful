try:
    from fpdf import FPDF

    has_fpdf: bool = True
except ImportError:
    has_fpdf: bool = False

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x  # Fallback to a no-op


class AutoReport(FPDF):
    def __init__(self):
        if not has_fpdf:
            raise Exception("ERROR: please, install fpdf for automatic report generation facilities")
        super().__init__()
        self.headings_font = 'Arial'
        self.body_text_font = 'Times'

    def box_width(self, string: str) -> float:
        return self.get_string_width(string) + 6

    def header(self) -> None:
        self.set_font(self.headings_font, 'B', 15)
        # *** Title ***
        w = self.box_width(self.title)
        # self.set_x((210 - w) / 2)
        self.set_text_color(114, 184, 81)
        self.cell(w, 6, self.title, 0, 0, 'L', False)
        # *** Author ***
        self.set_text_color(128)
        w = self.box_width(self.author)
        self.set_font(self.headings_font, 'I', 8)
        self.cell(w, 7, self.author, 0, 0, 'L', False)
        # *** Line break ***
        self.ln(10)

    def footer(self) -> None:
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font(self.headings_font, 'I', 8)
        # Text color in gray
        self.set_text_color(128)
        # Page number
        self.cell(0, 10, f'Page {self.page_no()} of {self.alias_nb_pages()}', 0, 0, 'R')

    def chapter_title(self, num: int, label: str) -> None:
        # Arial 12
        self.set_font(self.headings_font, '', 12)
        # Background color
        self.set_fill_color(210, 237, 197)
        # Title
        self.cell(0, 6, f'Section {num:d} : {label}', 0, 1, 'L', True)
        # Line break
        self.ln(4)

    def chapter_body(self, name: str) -> None:
        # Read text file
        with open(name, 'rb') as fh:
            txt = fh.read().decode('latin-1')
        self.set_font(self.body_text_font, '', 10)
        # Output justified text
        self.multi_cell(0, 5, txt)
        # Line break
        self.ln()

    def print_text(self, num: int, title: str, text: str) -> None:

        self.add_page()
        self.chapter_title(num, title)
        if text != "":
            self.chapter_body(text)

    def print_figures(self, num: int, title: str,
                      figures=None) -> None:
        if figures is None:
            figures = []

        self.add_page()
        self.chapter_title(num, title)
        if figures:
            vcounter: int = 0
            current_figure_in_page_index: int = 0
            current_fig_number: int = 1
            for fig in tqdm(figures, "Writing plots to report..."):
                # Get width of images
                image_width = 0.4 * self.w
                # Draw images side by side
                self.image(name=fig,
                           x=5 + self.x + ((image_width * 1) + 10 if current_figure_in_page_index % 2 == 1 else 0),
                           y=self.y + (image_width * vcounter),
                           w=image_width, h=0, type='')
                # Put figures side by side
                if current_figure_in_page_index % 2 == 1:
                    vcounter += 1
                # New page every 6 figures
                if current_figure_in_page_index == 5 and current_fig_number < len(figures):
                    current_figure_in_page_index = 0
                    vcounter = 0
                    self.add_page()
                else:
                    current_figure_in_page_index += 1
                current_fig_number += 1
