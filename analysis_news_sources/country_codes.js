rows = document.querySelector("#mw-content-text > div.mw-parser-output > table:nth-child(19)").querySelectorAll("tr");
country_codes = Array()
for (var i = 0; i < rows.length; i ++) { country_codes[rows[i].querySelector("a").title] = { "country_name" : rows[i].querySelectorAll("a")[1].innerText } }
delete country_codes[""]