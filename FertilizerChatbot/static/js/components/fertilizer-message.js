Vue.component('fertilizer-messages', {
    props: {
        messages: Array,
    },

    methods : {
		formatDate : function(dateVal) {
			var newDate = new Date(dateVal);
			var sMonth = this.padValue(newDate.getMonth() + 1);
			var sDay = this.padValue(newDate.getDate());
			var sYear = newDate.getFullYear();
			var sHour = newDate.getHours();
			var sMinute = this.padValue(newDate.getMinutes());
			var sAMPM = "AM";
		
			var iHourCheck = parseInt(sHour);
		
			if (iHourCheck > 12) {
				sAMPM = "PM";
				sHour = iHourCheck - 12;
			}
			else if (iHourCheck === 0) {
				sHour = "12";
			}
			sHour = this.padValue(sHour);
			return sDay + "-" + sMonth + "-" + sYear + " " + sHour + ":" + sMinute + " " + sAMPM;
		},
		
		padValue : function (value) {
			return (value < 10) ? "0" + value : value;
		},

        getMsgClass : function(message) {
            return (message.owner == 'BOT') ? "row border border-1 w-50 m-2 message-card-left" : "row border border-1 m-2 message-card-right";
        }
    },

    template: 
    `
    <div>
        <div :class=getMsgClass(message) v-for="message in messages">

            <div v-if="message.type == 'FORM'">
                <fertilizer-form :message="message"></fertilizer-form>
            </div>

            <div v-else-if="message.type == 'VIDEO'">
                <fertilizer-video :message="message"></fertilizer-video>
            </div>

            <div v-else class="col p-2">
                <b class="font-weight-bold" v-if="message.owner == 'BOT'">BOT : </b>
                <b class="font-weight-bold" v-else>You : </b>
                {{ message.content }}
            </div>
            <span class="text-secondary text-right fw-light">
                {{formatDate(message.createdon)}}
            </span>
        </div>
    </div>
    `,
});