---
layout: page
title: Seen Movies
permalink: /movies
exclude: true
---

I am trying to keep track of the movies I have watched.
The list is not (yet) exhaustive!

I use 👍 to denote a movie I enjoyed and reserve 🌶 for only the most sublime flicks.

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
