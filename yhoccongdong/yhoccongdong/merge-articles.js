import * as fs from "fs";

console.log("🔀 Merging old and new articles...");

// Đọc dữ liệu từ hai file
const oldArticles = JSON.parse(fs.readFileSync('articles.json', 'utf8'));
const newArticles = JSON.parse(fs.readFileSync('articles-updated.json', 'utf8'));

console.log(`📊 Old articles: ${oldArticles.length}`);
console.log(`📊 New articles: ${newArticles.length}`);

// Lọc bỏ các bài viết lỗi hoặc invalid từ file mới
const validNewArticles = newArticles.filter(article => 
  article.title && 
  article.content && 
  article.link &&
  article.title !== 'Error loading title' &&
  article.content !== 'Error loading content' &&
  article.content.length > 100 &&
  article.link.includes('/san-phu-khoa/')
);

console.log(`✅ Valid new articles: ${validNewArticles.length}`);

// Tìm những bài viết mới thực sự (không trùng với bài viết cũ)
const oldLinks = new Set(oldArticles.map(a => a.link));
const trulyNewArticles = validNewArticles.filter(article => !oldLinks.has(article.link));

console.log(`🆕 Truly new articles: ${trulyNewArticles.length}`);

if (trulyNewArticles.length > 0) {
  console.log('\n🎯 New articles found:');
  trulyNewArticles.forEach((article, index) => {
    console.log(`${index + 1}. ${article.title}`);
  });
}

// Merge dữ liệu: bài viết mới trước, bài viết cũ sau
const mergedArticles = [...trulyNewArticles, ...oldArticles];

console.log(`\n📈 Total articles after merge: ${mergedArticles.length}`);

// Lưu file kết quả
fs.writeFileSync('articles-complete.json', JSON.stringify(mergedArticles, null, 2));

console.log('✅ Merged data saved to articles-complete.json');

// Thống kê
const totalContentLength = mergedArticles.reduce((sum, a) => sum + (a.content ? a.content.length : 0), 0);
const avgContentLength = Math.round(totalContentLength / mergedArticles.length);

console.log(`\n📊 Final Statistics:`);
console.log(`   Total articles: ${mergedArticles.length}`);
console.log(`   New articles added: ${trulyNewArticles.length}`);
console.log(`   Average content length: ${avgContentLength} characters`);
console.log(`   Total content size: ${Math.round(totalContentLength / 1024)} KB`);
