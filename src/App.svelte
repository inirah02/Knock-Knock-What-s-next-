<script>

	import axios from "axios";

	import Icon from '@iconify/svelte';

	let word = ""
	let prediction = ""

	const handleClick = () => {
		//send axios request
		axios.post("http://localhost:5000/predictword", {data: word})
		.then((res)=>{
			word = word+" "+res.data.data.data;
		})
		//populate prediction
	}

	const handleInput = (e) => {
		//populates the word
		word = e.target.value;
	}

</script>

<main class="main-div">
	<div style="display: flex; flex-direction: column;">
		<div style="font-size: 32px; font-weight: 800; margin-bottom: 5vh;">knock, knock what's next?</div>
		<input type="text" on:input={handleInput} value={word} style="width: 160vh; height: 15vh; border-radius: 20vh; font-size: 64px; font-weight: 800; padding-bottom: 6vh; text-align: center;">
		<div style="display: flex; justify-content: center; font-size: 24px;"><button style="width: 30vh; border-radius: 10vh;" on:click={handleClick}><Icon icon="heroicons-outline:light-bulb" /><div>Predict!</div></button></div>
	</div>
</main>

<style>

</style>