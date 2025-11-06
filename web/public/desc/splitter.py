file = open("full.txt", "r")
content = file.read().split("% -----------------------------------------------------")
file.close()

for i, text in enumerate(content):
    target = open(f"{i + 1}_desc.tex", "w+")
    target.write(text)
    target.close()
