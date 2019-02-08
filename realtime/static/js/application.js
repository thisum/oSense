$(document).ready(function(){
    //connect to the socket server.
    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');
    var labels = [];
    var count = 0;
    var final_label = '';

    //receive details from server
    socket.on('newnumber', function(msg) {
        //maintain a list of ten numbers
        if (labels.length === 10){
            final_label = get_most_count(labels);
            $('#log').html('<span>' +  final_label + '</span>');
            labels = [0]
        }

        $('#posture').html('<span>' +  msg.posture + '</span>');
        $('#activity').html('<span>' +  msg.activity + '</span>');

        labels.push(msg.predict);
    });

    function get_most_count(array){
        var mostFrequent;
        var counts = {};
        var compare = 0;
        for(var i = 0, len = array.length; i < len; i++){
            var word = array[i];

            if(counts[word] === undefined){
               counts[word] = 1;
            }else{
               counts[word] = counts[word] + 1;
            }
            if(counts[word] > compare){
                 compare = counts[word];
                 mostFrequent = array[i];
            }
        }

        return mostFrequent;
    }

});