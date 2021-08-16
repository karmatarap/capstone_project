import asyncio
import csv
import sys
from requests_html import HTMLSession

def soundcloud_to_mp3(url, title):
	new_url = f'http://soundcloudtomp3.app/download/?url={url}'
	request = HTMLSession().get(new_url)
	request.html.render(keep_page=True, sleep=5)
	requests = []

	async def intercept(r):
		r._allowInterception = True
		requests.append(r)
		await r.continue_()
	request.html.page.on('request', lambda req: asyncio.ensure_future(intercept(req)))

	download_button = asyncio.get_event_loop().run_until_complete(request.html.page.querySelector('#dlMP3'))
	asyncio.get_event_loop().run_until_complete(download_button.click())
	asyncio.get_event_loop().run_until_complete(asyncio.sleep(2))

	audio_requests = [r for r in requests if r.response and r.response.headers.get('content-type') == 'audio/mpeg']
	audio_bytes = asyncio.get_event_loop().run_until_complete(audio_requests[0].response.buffer())

	audio_output = f'./elephantvoices/rumbles/{title}.mp3'
	with open(audio_output, 'wb') as f:
		f.write(audio_bytes)
	asyncio.get_event_loop().run_until_complete(request.html.page.close())

if __name__ == '__main__':
	input_path = './elephantvoices_labels.csv'
	counter = 0

	with open(input_path, 'r') as f:
		csv_reader = csv.reader(f)
		header = next(csv_reader)
		for row in csv_reader:
			url, title = row[0], row[1]
			soundcloud_to_mp3(url, title)
			counter += 1
			print(counter, title)
