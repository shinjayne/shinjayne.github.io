---
layout : post
title : jekyll의 Variables
date: 2017-05-24 22:40:30 +0900
comments : true
---


# **이 글이 다루는 것**
---


1. `jekyll` 의 전역 변수
2. `jekyll` 의 사이트 변수
3. `jekyll` 의 페이지 변수
4. 간단한 코드 (`liquid syntax`)





jekyll 로 나만의 개발 blog를 건설하는 중이다.
첫번째 게시글로 무엇이 좋을까 고민하다가 jekyll 의 사용법을 정리해보기로 했다.
그중에서도 이번에 살펴볼 내용은 **jekyll의 변수** 이다.

jekyll의 변수는 `html`문서나 `markdown` 으로 웹문서를 작성할 때 `liquid syntax`와 함께 유용하게 이용할 수 있다.     


# **jekyll의 전역 변수**
---

jekyll 변수의 큰 틀은 이렇다.

code | description
---|---
`site`|사이트의 정보 + `\_config.yml` 의 환경 설정 정보.
`page`|해당 페이지의 고유정보 + YAML 머리말 + (YAML 머리말에서 설정한 사용자 변수 )
`content`|`layout`에 포함된 `page`의 컨텐츠. (즉, `_layout` 폴더 안에 있는 파일들에서 주로 사용한다)
`paginator`|`\_config.yml`의 환경설정에 `paginate` 가 설정되어 있을 때 사용할 수 있다.


# **site**
---

code|description
---|---
`site.time`|current time(when you run `jekyll` in terminal)
`site.pages`|List of all pages
`site.posts`|List of all pages (arranged by older date)
`site.related_posts`| If the opening file is `post` , List of maximum 10 related posts of that file.
`site.static_files`|


# **page**
---

code|description
---|---
`page.content`|the content of the page
`page.title`|the title of the page
`page.excerpt`|the



<div id="disqus_thread"></div>
<script>

(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = '//{{ site.disqus.id }}.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
