import * as fs from "fs";
import * as cheerio from "cheerio";
import axios from "axios";

import ora from "ora";
import pLimit from "p-limit";
import consola from "consola";
import Emittery from "emittery";

import { normalizeParagraph } from "./lib/normalize-text.js";

const ArticlesFilePath = "./articles-updated.json";

const spinner = ora();
const limit = pLimit(3); // Giảm xuống 3 để tránh quá tải server
const emitter = new Emittery();
let completed = 0;

const ScrapeStartEvent = Symbol("ScrapeStartEvent");
const ScrapeCompleteEvent = Symbol("ScrapeCompleteEvent");
const ScrapeDoneEvent = Symbol("ScrapeDoneEvent");

// Hàm sleep để tránh quá tải server
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

async function extractArticle(link) {
  try {
    // Thêm delay để tránh quá tải server
    await sleep(100);
    
    const $ = await cheerio.fromURL(link);

    $(".lwptoc_toggle").remove();

    const { title, content } = $.extract({
      title: "h1",
      content: ".post-dt-content",
    });

    const sanitizedTitle = title ? title.trim() : '';
    const sanitizedContent = content ? normalizeParagraph(content) : '';

    return {
      title: sanitizedTitle,
      content: sanitizedContent,
      link,
    };
  } catch (error) {
    console.error(`Error extracting article ${link}:`, error.message);
    return {
      title: 'Error loading title',
      content: 'Error loading content',
      link,
    };
  }
}

// Hàm để lấy thêm bài viết từ AJAX endpoint
async function loadMoreArticles(page = 1) {
  try {
    console.log(`Trying to load page ${page}...`);
    
    // Thử gọi AJAX endpoint của WordPress
    const response = await axios.post(
      'https://yhoccongdong.com/san-phu-khoa/wp-admin/admin-ajax.php',
      new URLSearchParams({
        'action': 'loadMoreArchive',
        'page': page,
        'posts_per_page': 20
      }),
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'X-Requested-With': 'XMLHttpRequest',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
          'Accept': 'text/html, */*; q=0.01',
          'Accept-Language': 'vi,en;q=0.9',
          'Referer': 'https://yhoccongdong.com/san-phu-khoa/'
        },
        timeout: 10000
      }
    );

    if (response.data && response.data.trim() !== '') {
      const $ = cheerio.load(response.data);
      const links = [];
      
      $('.post-item_content-title').each((i, el) => {
        const href = $(el).attr('href');
        if (href) {
          links.push(href);
        }
      });
      
      console.log(`Page ${page}: Found ${links.length} articles`);
      return links;
    }
    
    console.log(`Page ${page}: No content returned`);
    return [];
  } catch (error) {
    console.log(`Page ${page}: Error - ${error.message}`);
    return [];
  }
}

// Hàm để thử các cách khác nhau để lấy toàn bộ articles
async function extractAllArticleLinks() {
  console.log("🔍 Extracting all article links...");
  let allLinks = [];

  // Bước 1: Lấy articles từ trang chính
  try {
    console.log("📄 Loading main page...");
    const $ = await cheerio.fromURL("https://yhoccongdong.com/san-phu-khoa/");
    const { links: mainPageLinks } = $.extract({
      links: [
        {
          value: "href",
          selector: ".post-item_content-title",
        },
      ],
    });
    
    allLinks = [...mainPageLinks];
    console.log(`✅ Main page: Found ${mainPageLinks.length} articles`);
  } catch (error) {
    console.error("❌ Error loading main page:", error.message);
    return [];
  }

  // Bước 2: Thử lấy thêm từ pagination
  let page = 2;
  let consecutiveEmptyPages = 0;
  const maxEmptyPages = 3;

  while (consecutiveEmptyPages < maxEmptyPages && page <= 50) { // Giới hạn tối đa 50 trang
    await sleep(1000); // Đợi 1 giây giữa các request
    
    const pageLinks = await loadMoreArticles(page);
    
    if (pageLinks.length > 0) {
      // Lọc duplicate
      const newLinks = pageLinks.filter(link => !allLinks.includes(link));
      allLinks = [...allLinks, ...newLinks];
      console.log(`📑 Page ${page}: Added ${newLinks.length} new articles (${pageLinks.length} total found)`);
      consecutiveEmptyPages = 0;
    } else {
      consecutiveEmptyPages++;
      console.log(`⚠️  Page ${page}: Empty (${consecutiveEmptyPages}/${maxEmptyPages})`);
    }
    
    page++;
  }

  // Bước 3: Thử scrape từ sitemap hoặc archive pages
  if (allLinks.length <= 50) {
    console.log("🗺️  Trying to find more articles from archive pages...");
    
    try {
      // Thử lấy từ trang archive theo tháng/năm
      const archiveUrls = [
        'https://yhoccongdong.com/san-phu-khoa/2024/',
        'https://yhoccongdong.com/san-phu-khoa/2023/',
        'https://yhoccongdong.com/san-phu-khoa/2025/'
      ];

      for (const archiveUrl of archiveUrls) {
        try {
          console.log(`📚 Checking archive: ${archiveUrl}`);
          const $ = await cheerio.fromURL(archiveUrl);
          const { links: archiveLinks } = $.extract({
            links: [
              {
                value: "href",
                selector: "a[href*='/san-phu-khoa/']",
              },
            ],
          });
          
          // Lọc chỉ lấy link bài viết (không phải trang category, etc.)
          const articleLinks = archiveLinks.filter(link => 
            link.includes('/san-phu-khoa/') && 
            !link.includes('/category/') &&
            !link.includes('/tag/') &&
            !link.includes('/page/') &&
            !allLinks.includes(link)
          );
          
          allLinks = [...allLinks, ...articleLinks];
          console.log(`📑 Archive ${archiveUrl}: Added ${articleLinks.length} articles`);
          
          await sleep(2000); // Đợi 2 giây giữa các archive request
        } catch (error) {
          console.log(`⚠️  Archive ${archiveUrl}: Error - ${error.message}`);
        }
      }
    } catch (error) {
      console.log("⚠️  Error accessing archive pages:", error.message);
    }
  }

  // Loại bỏ duplicate và invalid links
  const uniqueLinks = [...new Set(allLinks)].filter(link => 
    link && 
    link.startsWith('https://') && 
    link.includes('/san-phu-khoa/')
  );
  
  console.log(`🎯 Total unique articles found: ${uniqueLinks.length}`);
  return uniqueLinks;
}

// Main execution
async function main() {
  const links = await extractAllArticleLinks();

  if (links.length === 0) {
    console.error("❌ No articles found!");
    return;
  }

  emitter.on(ScrapeStartEvent, () => {
    consola.info("🚀 Starting updated scrape");
    spinner.start("Scraping articles...");
  });

  emitter.on(ScrapeCompleteEvent, () => {
    completed += 1;
    spinner.text = `Scraping articles: ${completed}/${links.length}`;
  });

  emitter.on(ScrapeDoneEvent, (articles) => {
    spinner.succeed(`✅ Scraping completed: ${links.length} articles scraped.`);
    consola.info("🏁 End");
    fs.writeFileSync(ArticlesFilePath, JSON.stringify(articles, null, 2));
    consola.box(`\n📁 Articles saved to ${ArticlesFilePath}`);
    
    // Thống kê
    const validArticles = articles.filter(a => a.title && a.content && a.content.length > 100);
    console.log(`\n📊 Statistics:`);
    console.log(`   Total articles: ${articles.length}`);
    console.log(`   Valid articles: ${validArticles.length}`);
    console.log(`   Average content length: ${Math.round(validArticles.reduce((acc, a) => acc + a.content.length, 0) / validArticles.length)} characters`);
  });

  const extractArticlePromises = links.map((l) =>
    limit(async () => {
      const article = await extractArticle(l);
      emitter.emit(ScrapeCompleteEvent);
      return article;
    })
  );

  emitter.emit(ScrapeStartEvent);
  const articles = await Promise.all(extractArticlePromises);
  emitter.emit(ScrapeDoneEvent, articles);
}

main().catch(console.error);
