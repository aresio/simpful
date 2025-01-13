from simpful.cluster_labeling import AutoReport


def main():
    pdf = AutoReport()
    pdf.set_title("Puberty Dataset Report")
    pdf.set_author('Automatically generated from PyFume data')
    pdf.print_figures(1, 'Features', figures=[f"plots/puberty/{x}.png" for x in range(11)])
    pdf.print_text(2, 'Rules', 'plots/puberty/rules.txt')

    pdf.output('test_report.pdf', 'F')


if __name__ == "__main__":
    main()
