import os
import asyncio
import aiofiles
import pandas as pd
import httpx
from parsel import Selector

url_list = [f'https://news.nankai.edu.cn/mtnk/system/count//0006000/000000000000/000/000/c0006000000000000000_000000{i}.shtml' for i in range(750,903)]
coroutine = asyncio.Semaphore(5)
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

url_dict = {}
url_df = pd.DataFrame(columns=['url'])
url_df.index.name = 'title'

async def parse_catalogs_page(url):
    async with coroutine:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            print(url)
            if response.status_code == 302:
                # 获取重定向的新 URL
                redirect_url = response.headers.get("Location")
                if redirect_url:
                    # 使用新的 URL 发起请求
                    response = await client.get(redirect_url)
            selector = Selector(response.text)
            print(response.text)
            url_dict.update(zip(selector.css('a::attr(href)').getall(), selector.css('a::text').getall()))


async def parse_page(url):
    async with coroutine:
        print("正在获取..."+str(url))
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=10,
                                         headers={'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36 Edg/119.0.0.0'}) as client:
                if url.startswith('http') or url.startswith('https'):
                    response = await client.get(url)
                    selector = Selector(response.text)
                    title = selector.css('title::text').get()
                    try:
                        if "/" in title:
                            title = title.replace("/", "_")
                        async with aiofiles.open(f'./nk_media/{title}.html', mode='w', encoding='utf-8') as file:
                            await file.write(response.text)
                        url_df.loc[title] = url  # 插入df
                    # 爬取的网页存在问题
                    except Exception as e:
                        print(f'{e}: {url}|{title}')
        except:
            print(f'error: {url}')


async def main():
    if not os.path.exists('./nk_media'):
        os.mkdir('./nk_media')

    tasks = [asyncio.create_task(parse_catalogs_page(url)) for url in url_list]
    await asyncio.gather(*tasks)

    tasks = [asyncio.create_task(parse_page(url)) for url in url_dict.keys()]
    await asyncio.gather(*tasks)

    # 写入文件
    url_df.to_csv("./nk_media.csv")


if __name__ == '__main__':
    asyncio.run(main())