Vue.component('fertilizer-form', {
    props: {
        message: Object,
    },

    methods : {
		generateOptions : function(options,value){
            let template = ``;
            for(let option of options) {
                if(option.value === value){
                    template+='<option value='+option.value+' selected>'+option.label+'</option>';
                } else {
                    template+='<option value='+option.value+'>'+option.label+'</option>';
                }
            }
            return template
        },

        submitSuggestion: function(event) {
            event.preventDefault();

            var form = jQuery(event.currentTarget);
            var formdata = {}
            
            for(let prop of form.serializeArray()) {
                formdata[prop.name] = prop.value.trim()
            }
            this.$root.submitForm({type : 'MULTI_FORM', data : formdata.tag});
        },
        submit: function(event) {
            event.preventDefault();

            var form = jQuery(event.currentTarget);
            var formdata = {}
            
            for(let prop of form.serializeArray()) {
                formdata[prop.name] = prop.value.trim()
            }
            this.$root.submitForm({type : 'SINGLE_FORM', data : formdata});
        },
    },

    template: 
    `
<div>
    <div class="border border-3 m-2 p-5" v-if="message.content.type == 'BASIC'">
        <h5 class="mt-4">{{ message.content.heading }}</h5>
        <form method="POST" @submit.prevent="submit">
            <div v-for="field in message.content.data">
                <div class="mt-3" v-if="field.type == 'text'">
                    <label :for=field.name class="form-label">{{ field.label }}</label>
                    <input :type=field.type class="form-control" :placeholder=field.placeholder :name=field.name :value=field.value :required=field.required>
                </div>
                <div class="mt-3" v-if="field.type == 'select'" :required=field.required>
                    <label :for=field.name class="form-label">{{ field.label }}</label>
                    <select :name=field.name class="form-select form-select-sm" v-html="generateOptions(field.options,field.value)"></select>
                </div>
            </div>
                <button type="submit" class="mt-3 mb-3 btn btn-primary">Submit</button>
        </form>
    </div>
    <div class="border border-3 p-2" v-if="['SUGGESTION','RELATED_QUESTIONS'].includes(message.content.type)">
        <h5 class="mt-4" v-if='message.content.type == "SUGGESTION"'>Did You Mean : </h5>
        <h5 class="mt-4" v-else>You may also want to look at : </h5>
        <div v-for="suggestion in message.content.data">
            <form class="" method="POST" @submit.prevent="submitSuggestion">
                <input type="hidden" name="tag" :value="suggestion.value" class="inputElement">
                <button type="submit" class="btn btn-primary btn-gradient m-1" >{{suggestion.label}}</button>
            </form>
        </div>
    </div>
</div>
    `,
});