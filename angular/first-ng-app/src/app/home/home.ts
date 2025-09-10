import { Component, signal } from '@angular/core';
import { Greeting } from '../components/greeting/greeting';
import { Counter } from '../components/counter/counter';

@Component({
  selector: 'app-home',
  imports: [Greeting, Counter],
  templateUrl: './home.html',
  styleUrl: './home.scss'
})
export class Home {
  toChildMessage = signal('Message To Child From Home');
  keyUpHandler(event: KeyboardEvent) {
    console.log(`${event.key}`);
  }
}
