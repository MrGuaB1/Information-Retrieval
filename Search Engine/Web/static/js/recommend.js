function query_suggest(keywords){
    let suggestion_dom = document.getElementById('suggestion');
    suggestion_dom.innerHTML = '<li>Loading suggestions...</li>';
    $.ajax({
        type : "GET",
        async: true,
        url : "/suggest?keywords=" + encodeURIComponent(keywords),
        success : function(data){
            let tag = '';
            for(let i = 0; i < data.length; i++){
                tag += `<li><a href="${data[i].url}" target="_blank">${data[i].text}</a></li>`;
            }
            suggestion_dom.innerHTML = tag;
        },
        error: function(){
            suggestion_dom.innerHTML = '<li>Error loading suggestions</li>';
        }
    });
}
function personal_suggest() {
    let suggestion_dom = document.getElementById('personal_suggestion');
    suggestion_dom.innerHTML = '<li>Loading suggestions...</li>';
        $.ajax({
        type : "GET",
        async: true,
        url : "/personalized_recommendation",
        success : function(data){
            let tag = '';
            for (let i = 0; i < data.length; i++) {
                let url = data[i][0];
                let name = data[i][1];
                if(name.length>20) {
                    name = name.substr(0,20);
                }
                name += "...";
                tag += `<li><a href="${url}" target="_blank">${name}</a></li>`;
            }
            suggestion_dom.innerHTML = tag;
            console.log(tag);
        },
        error: function(){
            suggestion_dom.innerHTML = '<li>Error loading suggestions</li>';
        }
    });
}
