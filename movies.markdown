---
layout: page
title: Seen Movies
permalink: /movies
exclude: true
---

I am trying to keep track of the movies and mini-series I have watched.
The list is not (yet) exhaustive!

I use 👍 to denote something I enjoyed and reserve 🌶 for my favourites.

<table>
  {% for row in site.data.movies %}
    {% if forloop.first %}
      <tr>
        <th>Name</th>
        <th>Year</th>
        <th>Rating</th>
      </tr>
    {% endif %}
      <tr>
        <td><a href="https://www.imdb.com/title/{{ row.imdb_id }}">{{ row.name }}</a></td>
        <td>{{ row.year }}</td>
        <td>{{ row.rating }}</td>
      </tr>
 {% endfor %}
</table>
