const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');

// Simple in-memory index (could use a DB like SQLite for persistence)
let index = { pages: {}, keywords: {} };

// Function to fetch and parse a page
async function crawlPage(url, maxPages = 10) {
    const visited = new Set();
    const toVisit = [url];
    let pagesCrawled = 0;

    while (toVisit.length > 0 && pagesCrawled < maxPages) {
        const currentUrl = toVisit.shift();
        if (visited.has(currentUrl)) continue;

        try {
            console.log(`Crawling: ${currentUrl}`);
            const response = await axios.get(currentUrl, { timeout: 5000 });
            const $ = cheerio.load(response.data);

            // Extract content
            const title = $('title').text() || 'No Title';
            const content = $('body').text().replace(/\s+/g, ' ').trim().slice(0, 1000);
            const links = [];
            $('a').each((i, elem) => {
                const href = $(elem).attr('href');
                if (href) {
                    const absoluteUrl = new URL(href, currentUrl).href;
                    if (absoluteUrl.startsWith('http')) links.push(absoluteUrl);
                }
            });

            // Index the page
            indexPage(currentUrl, title, content);
            visited.add(currentUrl);
            pagesCrawled++;
            toVisit.push(...links.filter(link => !visited.has(link)));
        } catch (error) {
            console.error(`Error crawling ${currentUrl}: ${error.message}`);
        }
    }

    // Save index to file (for persistence)
    fs.writeFileSync('index.json', JSON.stringify(index, null, 2));
}

// Index a page
function indexPage(url, title, content) {
    index.pages[url] = { title, content };

    // Simple keyword indexing
    const words = content.toLowerCase().split(' ').filter(w => w.length > 2);
    words.forEach(word => {
        if (!index.keywords[word]) index.keywords[word] = new Set();
        index.keywords[word].add(url);
    });
}

// Search function
function search(query) {
    const keywords = query.toLowerCase().split(' ').filter(w => w.length > 2);
    const matchingUrls = new Set();

    keywords.forEach(keyword => {
        if (index.keywords[keyword]) {
            index.keywords[keyword].forEach(url => matchingUrls.add(url));
        }
    });

    return Array.from(matchingUrls).map(url => ({
        url,
        title: index.pages[url].title,
        content: index.pages[url].content.slice(0, 200) + '...'
    }));
}

// Load existing index if available
if (fs.existsSync('index.json')) {
    index = JSON.parse(fs.readFileSync('index.json', 'utf8'));
}

module.exports = { crawlPage, search };