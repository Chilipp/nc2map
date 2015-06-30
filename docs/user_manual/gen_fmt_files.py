#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import argparse
import nc2map

parser = argparse.ArgumentParser(
    prog=__name__, usage='%s [options]' % __file__,
    description='Formatoption file generator')
parser.add_argument(
    '-table', default='content/fmt_keys.tex', metavar='<path-to-file>',
    help="Filename of the formatoption keyword table. Default: %(default)s")
parser.add_argument(
    '-list', default='content/fmt_docs.tex', metavar='<path-to-file>',
    help=(
        "Filename of the formatoption keyword description list. "
        "Default: %(default)s"))
parser.add_argument(
    '-gloss', default='content/fmt_gloss.tex', metavar='<path-to-file>',
    help=(
        "Filename of the formatoption glossary definitions. "
        "Default: %(default)s"))

all_keys = frozenset(nc2map.get_fmtkeys() + nc2map.get_fmtkeys('wind') +
                     nc2map.get_fmtkeys('simple'))

def main():
    args = parser.parse_args()
    with open(args.table, 'w') as f:
        keys = nc2map.get_fmtkeys()
        nkeys = 5  # number of formatoption keywords per row
        write_fmt_key_table(
            f, keys, 'List of formatoption keywords', 'fmt_keys', nkeys)
        keys = nc2map.get_fmtkeys('wind', 'windonly')
        write_fmt_key_table(
            f, keys, 'List of wind specific formatoption keywords',
            'windfmt_keys', nkeys)
        keys = nc2map.get_fmtkeys('simple')
        write_fmt_key_table(
            f, keys, 'List of LinePlot specific formatoption keywords',
            'simplefmt_keys', nkeys)

    with open(args.gloss, 'w') as f:
        for key in all_keys:
            f.write(
                '\\newglossaryentry{%s}{name=%s, type=index, '
                'description=\\nopostdesc}\n' % (
                    key, txt_to_tex(key, glsreplace=False)))

    groups = {
        '_general': "Basemap and axes formatoptions",
        '_cmapprops': "Colorbar and colormap formatoptions",
        '_windplotops': '',
        '_maskprops': 'Masking properties'}
    added_keys = set()
    remaining_keys = all_keys.copy()

    with open(args.list, 'w') as f:
        myfmt = nc2map.formatoptions.FieldFmt()
        patt = re.compile('(\n)(?![\n\n])')  # matches one single new line
        patt2 = re.compile(' {2,500}')
        for group, description in groups.items():
            if hasattr(myfmt, group):
                f.write('\\section{%s}\n' % description)
                f.write('\\begin{description}\n')
                for key in set(getattr(myfmt, group)).intersection(
                        remaining_keys):

                    doc = patt2.sub(
                        ' ', patt.sub('\\\\\\\\\n', patt.sub(
                            ' ', getattr(myfmt.__class__, key).__doc__)))
                    f.write('    \\item[%s:] \\label{item:%s} %s\n' % (
                        "\\gls*{%s}" % key, key, txt_to_tex(doc)))
                    remaining_keys -= {key}
                f.write('\\end{description}\n\n')

        myfmt = nc2map.formatoptions.WindFmt()
        f.write('\\section{Windplot specific formatoptions}\n')
        f.write('\\begin{description}\n')
        for key in sorted(set(myfmt.get_fmtkeys('windonly')).intersection(
                    remaining_keys)):
            doc = patt2.sub(  # replace multiple spaces
                ' ', patt.sub('\\\\\\\\\n', patt.sub(  # replace new lines
                    ' ', getattr(myfmt.__class__, key).__doc__)))
            f.write('    \\item[%s:] \\label{item:%s} %s\n' % (
                "\\gls*{%s}" % key, key, txt_to_tex(doc)))
            remaining_keys -= {key}
        f.write('\\end{description}\n\n')

        myfmt = nc2map.formatoptions.SimpleFmt()
        f.write('\\section{LinePlot specific formatoptions}\n')
        f.write('\\begin{description}\n')
        for key in sorted(set(myfmt.get_fmtkeys()).intersection(
                    remaining_keys)):
            doc = patt2.sub(  # replace multiple spaces
                ' ', patt.sub('\\\\\\\\\n', patt.sub(  # replace new lines
                    ' ', getattr(myfmt.__class__, key).__doc__)))
            f.write('    \\item[%s:] \\label{item:%s} %s\n' % (
                "\\gls*{%s}" % key, key, txt_to_tex(doc)))
            remaining_keys -= {key}
        f.write('\\end{description}\n\n')

        # write remaining keys
        f.write('\\section{Miscallaneous formatoptions}\n')
        f.write('\\begin{description}\n')
        for key in remaining_keys:
            try:
                doc = nc2map.get_fmtdocs()[key]
            except KeyError:
                doc = nc2map.get_fmtdocs('wind')[key]
            f.write('    \\item[%s:] \\label{item:%s} %s\n' % (
                "\\gls*{%s}" % key, key, txt_to_tex(doc)))
        f.write('\\end{description}\n\n')



def write_fmt_key_table(f, keys, caption, label, nkeys=5):
    """writes a table into the filehandler f with strings from keys and nkeys
    columns"""
    max_key_len = max(map(len, keys))*2 + len('\hyperref[item:]{}')
    f.write((
        '\\begin{table}\n'
        '    \\centering\n'
        '    \\captionabove{%s}\n'
        '    \\label{tab:%s}\n'
        '    \\begin{tabular}{%s|}\n') % (
            caption, label,
            ('|p{%1.2f\\textwidth}' % (1./nkeys-0.1/nkeys))*nkeys))
    f.write('        \\hline\n')
    for i in xrange(0,len(keys), nkeys):
        f.write(
            '        ' +
            ' & '.join(('\hyperref[item:%s]{%s}' % (
                key, txt_to_tex(key))).ljust(
                    max_key_len) for key in keys[i:i+nkeys]))
        if i + nkeys >= len(keys):
            f.write(' & '*(i + nkeys - len(keys)))
        f.write(' \\\\\n        \\hline\n')
    f.write((
        '    \\end{tabular}\n'
        '\\end{table}\n'))

def txt_to_tex(string, glsreplace=True):
    #if glsreplace:
        #for key in all_keys:
            #string = re.sub(r'\b%s\b' % key, "\\\\gls{%s}" % key, string)
            #string.replace(key, "\\gls{%s}" % key)
            #pass
    string = string.replace('_', '\\_')
    string = string.replace('#', '\\#')
    string = string.replace('%', '\\%')
    string = string.replace('{}', '$\lbrace\\rbrace$')
    string = re.sub("'(\w+)'", lambda x: '\\enquote{%s}' %
                    x.group().replace("'",""), string)
    string = string.replace('speed=sqrt(log(u)^2+log(v)^2)',
                            '$\mathrm{speed}=\\sqrt{'
                            '\\log(\\mathrm{u})^2+\\log(\\mathrm{v})^2}$')
    string = string.replace(
    nc2map.defaults.WindFmt['arrowstyle'],
    '$%s$' % nc2map.defaults.WindFmt['arrowstyle'])
    string = string.replace('<=', '$\leq$')
    string = string.replace('>=', '$\geq$')
    string = re.sub('\w<\w', lambda x: x.group().replace('<', '$<$'), string)
    string = re.sub('\w>\w', lambda x: x.group().replace('>', '$>$'), string)
    if ' - ' in string:
        ifirst = string.find(' - ')
        string = string[:ifirst] + '\n\\begin{itemize}' + \
            string[ifirst:] + '\n\\end{itemize}\n'
    if ' -- ' in string:
        ifirst = string.find(' -- ')
        ilast = string.find(' - ', string.rfind(' -- '))
        string = string[:ifirst] + '\n\\begin{itemize}' + \
            string[ifirst:ilast] + '\n\\end{itemize}\n' + \
                string[ilast:]
    string = string.replace(' - ', '\n    \\item ')
    string = string.replace(' -- ', '\n        \\item ')
    return string

if __name__ == '__main__':
    main()
