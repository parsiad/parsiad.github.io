---
permalink: /blog
title: Blog
---

<form method="get" id="search" action="https://duckduckgo.com">
  <input type="hidden" name="sites" value="https://parsiad.ca/blog">
  <input type="text" name="q" placeholder="Search&hellip;" autocomplete="off">
  <button type="submit">Search</button>
</form>

| Post | Date |
|---|---|
{% for post in site.posts %}| [{{ post.title }}]({{ post.url }}) | {{ post.date | date: "%B %d, %Y" }} |
{% endfor %}
