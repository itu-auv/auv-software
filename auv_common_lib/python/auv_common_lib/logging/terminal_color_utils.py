class TerminalColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    OKYELLOW = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ORANGE = "\033[38;5;214m"
    PASTEL_BLUE = "\033[38;5;110m"
    PASTEL_GREEN = "\033[38;5;157m"
    PASTEL_PURPLE = "\033[38;5;183m"

    @staticmethod
    def color_text(text, color):
        return f"{color}{text}{TerminalColors.ENDC}"

    @staticmethod
    def bold_text(text):
        return f"{TerminalColors.BOLD}{text}{TerminalColors.ENDC}"

    @staticmethod
    def underline_text(text):
        return f"{TerminalColors.UNDERLINE}{text}{TerminalColors.ENDC}"

    @staticmethod
    def rgb_text(text, r, g, b):
        return f"\033[38;2;{r};{g};{b}m{text}{TerminalColors.ENDC}"


# Example usage
# print(TerminalColors.color_text("This is a pastel blue text.", TerminalColors.PASTEL_BLUE))
# print(TerminalColors.color_text("This is an orange text.", TerminalColors.ORANGE))
# print(TerminalColors.rgb_text("This is a custom RGB color text.", 123, 200, 255))
# print(TerminalColors.bold_text("This is bold text."))
# print(TerminalColors.underline_text("This is underlined text."))
