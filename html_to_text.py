from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        print(f'  STARTTAG fired: <{tag}>  | skip={self.skip}')
        if tag in ('script', 'style', 'nav', 'footer'):
            self.skip = True
            print(f'    -> skip turned TRUE')

    def handle_endtag(self, tag):
        print(f'  ENDTAG fired: </{tag}>   | skip={self.skip}')
        if tag in ('script', 'style', 'nav', 'footer'):
            self.skip = False
            print(f'    -> skip turned FALSE')

    def handle_data(self, data):
        data = data.strip()
        if not data:
            return
        print(f'  DATA fired: \"{data}\"       | skip={self.skip}')
        if not self.skip:
            self.text.append(data)
            print(f'    -> APPENDED. text list now: {self.text}')
        else:
            print(f'    -> SKIPPED')

html = '''
<div>Hi Bhanu</div>
<nav>Skip this nav link</nav>
<div>This is visible text</div>
<script>console.log(skip this js)</script>
<p>Final paragraph</p>
'''

print('=== FEEDING HTML INTO PARSER ===')
print(f'HTML: {repr(html)}')
print()
p = TextExtractor()
p.feed(html)
print()
print('=== FINAL text list ===', p.text)
print('=== JOINED ===', ' '.join(p.text))
