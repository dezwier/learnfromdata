import re
import lfd
from lxml import etree as ET
import os
import inspect
import logging


def generate_doc(path='doc.html'):
    '''
    Get HTML documentation on the LFD package.
    
    Arguments
    ---------
    path : String, default 'doc.html'
            Path where the output documentation should be written. Should have an 
            .html extension.
    '''
    # Read from base HTML file
    with open(os.path.join(os.path.dirname(__file__), 'doc_base.html'), "r") as f: page = f.read()
    tree = ET.fromstring(page, parser=ET.HTMLParser())
    intro = tree.find(".//span[@id='intro']")
    content = tree.find(".//span[@id='content']")
    sidebar = tree.find(".//span[@id='sidebar']")
    
    # Title
    lfd_doc = h(f'Learnfromdata {span("v"+lfd.__version__, "small attributes")}', 1) + br()
    intro.addprevious(ET.XML(div(lfd_doc)))

    # Introduction tab
    sidebar.addprevious(ET.XML(pill('About', space=False, active=True)))
    content.addprevious(ET.XML(tab('Abouttab', inspect.getdoc(lfd), active=True)))

    # Collect objects to build a page for 
    transformers = [e.value for e in lfd.TransformEnum]
    models = [e.value for e in lfd.ModelEnum]
    plotters = [e.value for e in lfd.PlotterEnum]
    things = [lfd.Pipeline, lfd.Bootstrap, lfd.Data]
    things += transformers + models + plotters + [lfd.config]

    for thing in things:
        # Tab page
        class_intro = doc_function(thing)
        name = re.sub('lfd\.', '', thing.__name__)
        functions = get_functions(thing)
        doc = p(class_intro)
        doc += ''.join([doc_function(func) for func in functions])
        content.addprevious(ET.XML(tab(f'{name}tab', doc)))

        # Sidebar
        sub = thing in transformers[1:] + models[1:] + plotters[1:]
        sidebar.addprevious(ET.XML(pill(name, sub)))

    # Write to html file
    ET.ElementTree(tree).write(path, pretty_print=True, method="html")
    path = path if path.startswith('/') else os.path.join(os.path.abspath(os.getcwd()), path)
    logging.info(f'Doc available at {path}')


def doc_function(thing):
    '''
    Get HTML snippet with documentation on any python object.
    
    Arguments
    ---------
    thing : any object, like a function or class
            Docstring of this object will be parsed into HTML.
    '''
    if not callable(thing): return ''
    try:

        # Retrieve function, signature and docstring
        func_name = re.sub('\.__init__', '', thing.__qualname__)
        func_name = thing.__qualname__
        signature = str(inspect.signature(thing))
        docstring = make_html_proof(inspect.getdoc(thing).strip())
        sourcecode = make_html_proof(inspect.getsource(thing))
        sourcecode = re.sub("(\"\"\"|''')[\s\S]*?(\"\"\"|''')", "", sourcecode)

        # Docstring
        split_index = re.search('Arguments|Attributes', docstring)
        if split_index:
            split_index = split_index.start()
            description = docstring[:split_index].strip()
            arguments = docstring[split_index:]
            arguments = re.sub('-{2,}', '', arguments).strip()
            lines = [l.strip() for l in arguments.split('\n') if l.strip()]
            params = '<p>'
            for l, line in enumerate(lines):
                if ' : ' in line:
                    if 0 < l < len(lines) : params += br()
                    argument, datatype = line.split(" : ")
                    params += f'\n{span(b(argument), "argument")}: {span(i(datatype), "datatype")} '
                elif line.startswith('Arguments'): params += span(line, 'attributes')
                elif line.startswith('Attributes'): params += br()*2 + span(line, 'attributes')
                else: params += line + ' '
            params += '</p>\n'
        else: 
            description = docstring.strip()
            params = ''

        # Wrap in HTML
        function = f'<span class="function">{func_name}</span>'
        func_name = re.sub('\.', '', func_name) # Strip dots
        sourceicon = f'<i type="button" class="bi bi-code-slash" data-bs-toggle="modal" data-bs-target="#{func_name}mod"></i>'
        sourcecode = modal(func_name, sourcecode)
        signature = f'<span class="signature">{signature} {sourceicon}</span>'
        signature = h(function + signature, 3)
        description = p(description)
        return signature + description + params + sourcecode + br()
    except:
        logging.warning(f'Could not parse docstring for {thing}')
        return ''


def make_html_proof(string):
    string = re.sub('<', '&lt;', string)
    string = re.sub('>', '&gt;', string)
    string = re.sub('&', '&amp;', string)
    string = re.sub('`([^`]*)`', r'<pre><code class="python">\1</code></pre>', string)
    return string
def h(content, n): return f'<h{n}>{content}</h{n}>'
def p(content): return f'<p>{content}</p>'
def b(content): return f'<b>{content}</b>'
def i(content): return f'<i>{content}</i>'
def div(content): return f'<div>{content}</div>'
def span(content, classes=None): return f'<span class="{classes}">{content}</span>'
def br(): return '<br></br>'
def hr(): return '<hr></hr>'
def tab(identifier, content, active=False): 
    return f'<div class=\'tab-pane{" show active" if active else ""}\' \
        id="{identifier}" role="tabpanel" aria-labelledby="{identifier}" tabindex="0">{content}</div>'
def pill(name, sub=False, space=False, active=False):
    space = "style='margin-bottom:40px'" if space else ""
    active = " active" if active else ""
    sub = " sub" if sub else ""
    return f'<li class="nav-item" role="presentation"><button class="nav-link{active}{sub}" {space} data-bs-toggle="pill" \
        data-bs-target="#{name}tab" role="tab" type="button"  aria-controls="{name}tab" aria-selected="{"true" if active else "false"}">{name}</button></li>'
def modal(identifier, content):
    return f'''\n
    <div class="modal fade bd-example-modal-lg" id="{identifier}mod" tabindex="-1" role="dialog" aria-labelledby="{identifier}Label" aria-hidden="true">
        <div class="modal-dialog modal-xl" role="document">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">{identifier}</h5>
                    <button type="button" class="btn-close" data-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <pre><code class="python">{content}</code><br></br></pre>
                </div>
            </div>
        </div>
    </div>'''

def get_functions(thing):
    return [getattr(thing, name) for name in thing.__dict__ \
        if (not name.startswith('_')) and \
            callable(getattr(thing, name))]

def get_lfd_doc():
    """Get docstring for LFD module."""
    doc = ''
    path = os.path.dirname(__file__)
    with open(os.path.join(path, 'doc_lfd.html'), "r") as f: 
        doc += f.read()
    return doc


if __name__ == "__main__":
    generate_doc('doc.html')