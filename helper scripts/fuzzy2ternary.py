
doc2pos = {}
with open("output/tfidf-context-fuzzy-3.txt", encoding='utf8') as f:
    for line in f:
        line = line.strip()
        if '[' not in line:
            continue
        fields = line.split('[')
        doc2pos[fields[0].strip()] = fields[1][:-1].split()

doculects_tex = {'Dutch_Std_BE': 'Std. Dutch (BE)',
                 'Dutch_Std_NL': 'Std. Dutch (NL)',
                 'Graubuenden': 'Graub\\"{u}nden',
                 'Tuebingen': 'T\\"{u}bingen',
                 'Veenkolonien': 'Veenkoloni\\"{e}n'}

with open("doc/figures/fuzzy-ternary.tex", 'w', encoding='utf8') as f:
    f.write("\\documentclass[dvipsnames]{standalone}\n\\usepackage{pgfplots}\n")
    f.write("\\usepgfplotslibrary{ternary}\n\\usetikzlibrary{shapes}\n\\begin{document}\n")
    f.write("\\definecolor{purple}{HTML}{520066}\n")
    f.write("\\definecolor{blue}{HTML}{31688e}\n")
    f.write("\\definecolor{green}{HTML}{35b779}\n")
    f.write("\\begin{tikzpicture}\n")
    f.write("\t\\tikzstyle{ingv}=[circle, green, draw, inner sep = 2pt, fill=white]\n")
    f.write("\t\\tikzstyle{dutch}=[ingv, fill=green]\n")
    f.write("\t\\tikzstyle{central}=[ingv, rectangle, blue, fill=blue]\n")
    f.write("\t\\tikzstyle{upper}=[ingv, regular polygon, regular polygon sides=3, purple, fill=purple, inner sep = 1pt]\n")    
    f.write("\t\\begin{ternaryaxis}[\n")
    f.write("\t\txmin = 0, xmax = 1,\n\t\tymin = 0, ymax = 1,\n")
    f.write("\t\tzmin = 0, zmax = 1,\n")
    f.write("\t\txlabel = A, ylabel = B, zlabel = C,\n")
    f.write("\t\tgrid = both, label style = {sloped},\n")
    f.write("\t\tminor tick num = 3,\n\t]\n")
    f.write("\t\\addplot3[only marks, mark options={blue}] table {\n")
    for pos in doc2pos.values():
        f.write("\t\t" + '\t'.join(pos) + "\n")
    f.write("\t};\n")
    for doc, pos in doc2pos.items():
        f.write("\t\\node[ingv, label=below:")
        f.write(doculects_tex.get(doc, doc))
        f.write("] at (axis cs:" + ', '.join(pos) + ") {};\n")
    f.write("\t\\end{ternaryaxis}\n\\end{tikzpicture}\n\\end{document}")
