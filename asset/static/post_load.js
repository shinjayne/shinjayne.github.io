$(window).on('load', function () {
    load('#load-list', '10');
    $("#load-button").on("click", function () {
        load('#load-list', '5', '#load-button');
    })
});

function load(id, cnt, btn) {
    var unshowed = id + " .load-thing:not(.active)";
    var unshowed_length = $(unshowed).length;
    var how_many_new_loads;
    if (cnt < unshowed_length) {
        how_many_new_loads = cnt;
    } else {
        how_many_new_loads = unshowed_length;
        $(btn).hide();
    }
    $(unshowed + ":lt(" + how_many_new_loads + ")").addClass("active");
}
