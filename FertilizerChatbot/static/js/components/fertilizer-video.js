Vue.component('fertilizer-video', {
    props: {
        message: Object,
    },
    data: function () {
        return {
            iframeHTML : "",
        }
    },
    created() {
        this.iframeHTML = '<iframe width="420" height="345" src="'+this.message.content+'"></iframe>';
    },

    template: 
    `
    <div v-html="iframeHTML"></div>
    `,
});