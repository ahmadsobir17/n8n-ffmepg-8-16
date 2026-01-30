const fs = require('fs');
const data = JSON.parse(process.stdin.readAll());
console.log(JSON.stringify(data.nodes.find(n => n.name === "Basic LLM Chain"), null, 2));
