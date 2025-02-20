const chatbotApp = new Vue({
    el: '#app',
    data: function () {
        return {
            messages: [],
            input: ""
        }
    },
    methods: {

        getMessageType: function (type) {
            if (type == 'message') {
                return 'MESSAGE';
            } else if (type == 'video') {
                return 'VIDEO'
            } else if (type == 'relatedFAQ') {
                return 'RELATED_QUESTIONS'
            } else {
                return 'FORM'
            }
        },

        buildMessage: function (data) {
            let msgs = []
            if (data.tag !== 'invalid_text') {
                for (const [type, response] of Object.entries(data.prediction)) {
                    let TYPE = this.getMessageType(type);
                    if(TYPE !== 'RELATED_QUESTIONS') {
                        msgs.push(
                            { id: Date.now(), content: response, type: TYPE, owner: "BOT", createdon: new Date() }
                        )
                    } else {

                        let relatedQuestions = [];
                        for (const relatedQuestion of data.prediction.relatedFAQ) {
                            relatedQuestions.push({ label: relatedQuestion.label, value: relatedQuestion.tag, });
                        }

                        msgs.push(
                            { id: Date.now(), content: {
                                type: TYPE,
                                data: relatedQuestions
                            }, type: "FORM", owner: "BOT", createdon: new Date() }
                        )
                    }
                }
            } else {
                let suggestion = [];
                for (const allPrediction of data.all_predictions) {
                    suggestion.push({ label: allPrediction.label, value: allPrediction.tag, });
                }
                msgs.push(
                    { id: Date.now(), content: data.prediction.message, type: "MESSAGE", owner: "BOT", createdon: new Date() }
                )
                msgs.push(
                    { id: Date.now(), content: {
                        type: "SUGGESTION",
                        data: suggestion
                    }, type: "FORM", owner: "BOT", createdon: new Date() }
                )
            }
            return msgs;
        },

        send: function () {
            if (!this.input) { return false; }

            var thisInstance = this;
            let params = {
                language: 'eng',
                inputSentence: this.input
            }
            this.messages.push(
                { id: Date.now(), content: this.input, owner: "VISITOR", type: 'MESSAGE', createdon: new Date() }
            )
            this.input = '';
            $.get('/api/predict', params, function (data) {
                let msgs = thisInstance.buildMessage(data);
                thisInstance.messages = thisInstance.messages.concat(msgs);
            });
        },

        submitForm: function (form) {
            var thisInstance = this;
            if(form.type === 'MULTI_FORM') {
                let params = {
                    language: 'eng',
                    tag: form.data
                }
                $.get('/api/predict', params, function (data) {
                    let msgs = thisInstance.buildMessage(data);
                    thisInstance.messages = thisInstance.messages.concat(msgs);
                });
            } else {  
                form.data.language  = 'eng';
                $.ajax({
                    type: 'POST',
                    url: '/api/form/',
                    data: JSON.stringify(form.data),
                    contentType: "application/json; charset=utf-8",
                    traditional: true,
                    success: function (response) {
                        thisInstance.messages.push(
                            { id: Date.now(), content: response.message, owner: "BOT", type: 'MESSAGE', createdon: new Date() }
                        )
                    }
                });          

            }
        }
    },

    template:
    `
    <div class="">

        <div class="border" style="overflow: auto; height: 70vh;">
            <fertilizer-messages :messages="messages"></fertilizer-messages>
        </div>

        <div class="input-group mt-3 mb-5">
        <input type="text" v-model=input class="form-control" @keyup.enter="send" placeholder="Text Here..." aria-describedby="basic-addon2">
            <span class="input-group-text" id="basic-addon2" @click="send" v-on:keyup.enter="onEnter">Send</span>
        </div>
    </div>
    `,
});