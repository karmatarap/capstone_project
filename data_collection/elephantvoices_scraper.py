import csv
from requests_html import HTMLSession

def extract_from_page(url):
    request = HTMLSession().get(url)
    request.html.render(sleep=2)
    title = request.html.find('.soundTitle__title', first=True).text
    desc = request.html.find('.sc-type-small', first=True).text
    desc = f'"{desc}"'
    return title, desc

def get_all_urls(input_csv):
    urls = list(csv.reader(open(input_csv)))
    return urls

def extract_from_all_urls(urls):
    output = []
    for url in urls:
        title, desc = extract_from_page(url[0])
        output.append([url[0], title, desc])
    return output

def save_output(output, output_file):
    with open(output_file, 'wt') as f_out:
        f_out.write('url,title,desc\n')
        for line in output:
            f_out.write(','.join(line) + '\n')

if __name__ == '__main__':
    soundcloud_links = '/datasets/dzanga-bai/dzanga-bai/soundcloud_links.csv'
    output_file = '/datasets/dzanga-bai/dzanga-bai/elephantvoices_labels.csv'
    
    urls = get_all_urls(soundcloud_links)
    output = extract_from_all_urls(urls)
    save_output(output, output_file)