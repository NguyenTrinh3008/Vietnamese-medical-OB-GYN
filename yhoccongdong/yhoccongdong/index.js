import * as fs from "fs";
import * as cheerio from "cheerio";

import ora from "ora";
import pLimit from "p-limit";
import consola from "consola";
import Emittery from "emittery";

import { normalizeParagraph } from "./lib/normalize-text.js";

const ArticlesFilePath = "./articles.json";
const HTMLFilePath = "./assets/index.html";

const spinner = ora();
const limit = pLimit(5);
const emitter = new Emittery();
let completed = 0;

const ScrapeStartEvent = Symbol("ScrapeStartEvent");
const ScrapeCompleteEvent = Symbol("ScrapeCompleteEvent");
const ScrapeDoneEvent = Symbol("ScrapeDoneEvent");

const links = await extractArticleLinks();

emitter.on(ScrapeStartEvent, () => {
  consola.info("Start");
  spinner.start("Scraping articles...");
});

emitter.on(ScrapeCompleteEvent, () => {
  completed += 1;
  spinner.text = `Scraping articles: ${completed}/${links.length}`;
});

emitter.on(ScrapeDoneEvent, (articles) => {
  spinner.succeed(`Scraping completed: ${links.length} articles scraped.`);
  consola.info("End");
  fs.writeFileSync(ArticlesFilePath, JSON.stringify(articles, null, 2));
  consola.box(`\nArticles saved to ${ArticlesFilePath}`);
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

async function extractArticle(link) {
  const $ = await cheerio.fromURL(link);

  $(".lwptoc_toggle").remove();

  const { title, content } = $.extract({
    title: "h1",
    content: ".post-dt-content",
  });

  const sanitizedTitle = title.trim();
  const sanitizedContent = normalizeParagraph(content);

  return {
    title: sanitizedTitle,
    content: sanitizedContent,
    link,
  };
}

async function extractArticleLinks() {
  const buffer = fs.readFileSync(HTMLFilePath);
  const $ = cheerio.loadBuffer(buffer);
  const { links } = $.extract({
    links: [
      {
        value: "href",
        selector: ".post-item_content-title",
      },
    ],
  });
  return Promise.resolve(links);
}
